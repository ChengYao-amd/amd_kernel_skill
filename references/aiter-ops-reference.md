# AITER（AMD Inference Toolkit for Efficient Runtime）算子与工程参考

**AITER**（仓库名多为 `aiter`）是 AMD 侧 **集中维护的高性能 AI 算子库**：底层实现可来自 **Triton、Composable Kernel（CK）、inline assembly** 等；面向 **Python / C++ API**，既覆盖 **推理**，也包含 **训练** 与 **GEMM + 通信** 等组合场景。集成方通常把 AITER 作为 **统一算子来源**，再接入自有框架。

本文档按 **源码与配置布局** 归纳 **算子类别、精度与架构差异、调参入口**，便于在自定义 kernel 之前评估是否已有成熟实现。

## 1. 仓库与运行方式（官方 README 摘要）

- **获取**：`git clone --recursive https://github.com/ROCm/aiter.git`（子模块含第三方与内核依赖）。
- **安装**：`python3 setup.py develop` 或 `pip install -e .`。
- **可选**：**FlyDSL** 用于部分 **混合精度 MoE**（如 **A4W4**）；未安装时可回退 **CK** 路径。**Iris** 用于 **Triton 通信**（见 `requirements-triton-comms.txt` / `docs/triton_comms.md`）。
- **验证**：`python3 op_tests/test_<op>.py`（具体脚本以 `op_tests/` 目录为准）。

下列 **API 片段仅为说明层次**；真实函数名与参数请查阅 **`aiter` Python 包导出** 与 **`op_tests`** 中的用法。

```python
# 层次示意：以仓库内实际导出为准
# import aiter
# out = aiter.<op_name>(tensors..., **kwargs)
```

## 2. 算子分类（从模块与命名归纳）

以下为从 **目录与命名** 可系统检索到的 **操作族**；同一族内常有 **fwd/bwd**、**varlen**、**量化位宽**、**Gluon / ragged** 等变体。

| # | 类别 | 覆盖内容（关键词） |
|---|------|-------------------|
| 1 | **MHA / Flash Attention** | `mha_fwd` / `mha_bwd`，`fmha_v3_fwd` / `fmha_v3_bwd`，**varlen**，**fp8_pertensor**，**batch_prefill**；实现可组合 **CK + Triton**。 |
| 2 | **GEMM** | `gemm_a16w16`，`gemm_a8w8`（**asm / CK / tune**），`gemm_a8w8_blockscale`，`gemm_a4w4`，**batched_gemm**，**deepgemm** 等。 |
| 3 | **Paged Attention（PA）** | `pa_fwd_naive` / **asm**，`paged_attention_v1` / **ragged**，**Gluon decode** 相关路径。 |
| 4 | **Normalization** | `layer_norm`，`layernorm2d`，`rms_norm`，`rmsnorm2d`；常与 **add**、**smoothquant**、**dynamicquant** 等融合。 |
| 5 | **Activation** | `silu_and_mul`，`scaled_silu_and_mul`，`gelu_and_mul`，`gelu_tanh_and_mul`，`gelu_fast`。 |
| 6 | **Quantization** | `per_tensor_quant`，`per_token_quant`，`smoothquant`，`dynamic_per_tensor`，**FP4**（如 **per_1x32_f4_quant**），**block quant**，**MXFP4**。 |
| 7 | **MoE** | `fmoe`，`fmoe_g1u1`，`fmoe_fp8_blockscale_g1u1`，`moe_stage1` / `moe_stage2`，`moe_fused_gate`，`moe_sorting`，`ck_moe_stage1` / `ck_moe_stage2`。 |
| 8 | **RoPE** | `rope_fwd` / `rope_bwd`，`rope_cached`，`rope_2d`。 |
| 9 | **KV / Cache / MLA** | `reshape_and_cache`，`concat_and_cache_mla`，`fused_qk_rmsnorm_group_quant` 等。 |
| 10 | **Collective / Communication** | `all_reduce`，`reduce_scatter`，`all_gather`，`custom_all_reduce`，`quick_all_reduce`（部分依赖 **Triton + Iris** 栈）。 |
| 11 | **Sampling** | `greedy_sample`，`random_sample`，`topk_softmax`，`top_k_per_row`。 |
| 12 | **Elementwise** | `add`，`sub`，`mul`，`div`，`sigmoid`，`tanh` 等。 |

**README 中的功能表**（MHA、MLA、PA、FusedMoe、QUANT、RMSNORM、LAYERNORM、ROPE、GEMM 等）与上表一致，可作为 **对外沟通** 的简版索引。

## 3. FP8 与架构（gfx950 / gfx1250 vs 其它）

AITER 在 **FP8** 路径上区分 **OCP** 与 **非 OCP（FNUZ）** 变体（具体类型名以源码为准）：

| 目标架构（示例） | FP8 取向 |
|------------------|----------|
| **gfx950**、**gfx1250** | 使用 **E4M3FN**、**E5M2** 等 **OCP** 对齐格式。 |
| **其它 arch** | 常见 **E4M3FNUZ**、**E5M2FNUZ** 等 **FNUZ** 变体。 |

移植算子或对比 **MI300 vs MI355** 时，应先确认 **dtype 枚举与 tensor layout** 与目标 **ISA / OCP 支持** 一致，再谈性能。

## 4. 调优与配置基础设施

| 类型 | 位置（惯例） | 用途 |
|------|----------------|------|
| **CSV** | `aiter/configs/*.csv` | **按模型 / 场景** 调好的 **GEMM** 等参数表。 |
| **JSON** | `aiter/ops/triton/configs/**/*.json` | **Triton kernel** 的 block、split 等 **JSON 配置**。 |
| **MoE 分档** | 逻辑层常按 **token 规模 M** 分档 | 例如 **small_M**（M 小于 256）、**medium_M**（M 小于 1024）、**large_M**：对应不同专家并行与 kernel 选择。 |

## 5. Triton GEMM 配置参数（常见字段）

在 JSON 或 Python 配置中，Triton GEMM 常出现下列 **可调字段**（名称以具体文件为准）：

| 参数 | 作用 |
|------|------|
| `BLOCK_SIZE_M` / `BLOCK_SIZE_N` / `BLOCK_SIZE_K` | Tile 形状。 |
| `GROUP_SIZE_M` | M 向 **分组**，影响 L2 局部性与 wave 调度。 |
| `NUM_KSPLIT` / `SPLITK_BLOCK_SIZE` | **K 维切分** 与 **split-K** 块大小，用于大 K 或数值策略。 |
| `cache_modifier` | 全局 load 的 **cache hint**（如 `ca`、`cg`），与架构及共栖 kernel 有关。 |

## 6. 何时用 AITER、何时写自定义 kernel

| 场景 | 建议 |
|------|------|
| **标准算子 + 常见 shape + 已支持 dtype** | 优先 **AITER**，并用 **op_tests** 与业务 case 做回归。 |
| **库中无覆盖的融合 或 特殊 mask/layout** | **自定义 kernel**（CK / Triton / asm），并以 **AITER 最接近子算子** 为性能下界。 |
| **性能不达标** | 先对照 **CSV/JSON** 是否与当前 **模型 tier、arch、batch** 匹配，再考虑自定义。 |
| **罕见精度组合** | 先查 **FP8 / INT4 / MXFP4** 分支；若无，则自定义并注意上文 **架构差异** 小节。 |

## 7. Benchmark 对比（示意）

若仓库提供 benchmark 脚本，可采用与官方 **README / docs** 一致的入口；以下为 **模式化** 命令占位：

```bash
# 以仓库实际脚本为准，例如：
# python scripts/benchmark_kernel.py --kernel my_kernel.py --op attention --baseline aiter
```

### DeepSeek-R1 / V3 系列上的 AITER 加速比（公开博文，MI300X）

下列为相对未启用 **AITER** 基线的量级（具体模型与批次以原文为准）：

| 算子 / 场景 | Speedup |
|-------------|---------|
| **MLA decode** | **17×** |
| **Block-scale fused MoE** | **3×** |
| **Block-scale GEMM** | **2×** |
| **MHA prefill** | **14×** |
| **End-to-end（SGLang，8×MI300X）** | **2.1×**（例如 **6484 → 13704 tok/s**） |

## 8. 与 SGLang / vLLM 集成及 PD 分离

### SGLang

```bash
export SGLANG_USE_AITER=1
```

可与 **FlyDSL MoE** 等组合（示例）：`AITER_USE_FLYDSL_MOE=1` 及 `--disable-radix-cache --enable-torch-compile` 等，以仓库与场景为准。

### vLLM（ROCm）

总开关：

```bash
export VLLM_ROCM_USE_AITER=1
```

在总开关开启时，下列子开关常用于细粒度裁剪（默认值随版本变化，以 **vLLM** 文档为准）：

| 环境变量 | 作用 |
|----------|------|
| `VLLM_ROCM_USE_AITER_LINEAR` | **Linear** 层 **FP8** 量化 **GEMM** |
| `VLLM_ROCM_USE_AITER_MOE` | 融合 **MoE** 路由与计算 |
| `VLLM_ROCM_USE_AITER_RMSNORM` | **RMSNorm** 加速 |
| `VLLM_ROCM_USE_AITER_MLA` | **MLA**（**DeepSeek** 等） |
| `VLLM_ROCM_USE_AITER_MHA` | **MHA**（**Llama/Mistral** 等） |
| `VLLM_ROCM_USE_AITER_FP8BMM` | **MLA** 用 **FP8** **batched matmul** |
| `VLLM_ROCM_USE_SKINNY_GEMM` | 小 **batch** **skinny GEMM** |

**DeepSeek MLA** 使用 **vLLM** 时常需 **`--block-size 1`**，否则可能报错。

### Disaggregated serving（Prefill / Decode 分离）

**SGLang** 支持 **PD disaggregation**，将 **prefill** 与 **decode** 分到不同进程/**GPU**。典型 **launch_server** 相关参数包括：

| 参数 | 含义 |
|------|------|
| `--disaggregation-mode` | `prefill` 或 `decode` |
| `--disaggregation-transfer-backend` | **KV** 传输后端（如 **mooncake**、**nixl**） |
| `--disaggregation-ib-device` | **InfiniBand** / **RoCE** 设备名 |
| `--disaggregation-bootstrap-port` | **Bootstrap** 端口（爬取默认 **8998**） |
| `--base-gpu-id` | 多卡上起始 **GPU** 索引（**decode** 侧常用） |
| `--model-path` | 模型路径（两侧均需） |
| `--port` / `--host` | 服务监听地址与端口 |

前端常配合 **`python -m sglang_router.launch_router --pd-disaggregation --prefill ... --decode ...`**。细粒度线程与超时见 **`SGLANG_DISAGGREGATION_*`** 等环境变量（**thread pool**、**queue**、**heartbeat** 等）。

## 9. 与 CK 文档的衔接

- **CK tile / pipeline / GemmConfig\*** 细节：[[ck-tile-tuning.md|ck-tile-tuning]]、[[ck-programming-model.md|ck-programming-model]]
- AITER 中大量 GEMM / FMHA 仍依赖 **CK** 实现；阅读 **ck_tile** 有助于理解 **调参边界与瓶颈类型**。
