# AITER (AMD Inference Toolkit for Efficient Runtime) Operator & Engineering Reference

**AITER** (repository commonly named `aiter`) is AMD's **centrally maintained high-performance AI operator library**: underlying implementations may come from **Triton, Composable Kernel (CK), inline assembly**, etc.; exposing **Python / C++ APIs** that cover **inference**, as well as **training** and combined scenarios such as **GEMM + communication**. Integrators typically adopt AITER as a **unified operator source** and plug it into their own frameworks.

This document organizes **operator categories, precision and architecture differences, and tuning entry points** according to the **source code and configuration layout**, to help evaluate whether a mature implementation already exists before writing a custom kernel.

## 1. Repository & Execution (Official README Summary)

- **Obtain**: `git clone --recursive https://github.com/ROCm/aiter.git` (submodules include third-party and kernel dependencies).
- **Install**: `python3 setup.py develop` or `pip install -e .`.
- **Optional**: **FlyDSL** is used for some **mixed-precision MoE** (e.g., **A4W4**); falls back to the **CK** path when not installed. **Iris** is used for **Triton communication** (see `requirements-triton-comms.txt` / `docs/triton_comms.md`).
- **Verify**: `python3 op_tests/test_<op>.py` (refer to the `op_tests/` directory for specific scripts).

The **API snippets below are for illustration of the hierarchy only**; for actual function names and parameters, consult the **`aiter` Python package exports** and usage in **`op_tests`**.

```python
# Hierarchy illustration: refer to actual exports in the repository
# import aiter
# out = aiter.<op_name>(tensors..., **kwargs)
```

## 2. Operator Categories (Derived from Module & Naming Conventions)

The following are **operation families** systematically discoverable from **directories and naming**; within the same family there are often variants such as **fwd/bwd**, **varlen**, **quantization bit-widths**, **Gluon / ragged**, etc.

| # | Category | Coverage (Keywords) |
|---|----------|---------------------|
| 1 | **MHA / Flash Attention** | `mha_fwd` / `mha_bwd`, `fmha_v3_fwd` / `fmha_v3_bwd`, **varlen**, **fp8_pertensor**, **batch_prefill**; implementations can combine **CK + Triton**. |
| 2 | **GEMM** | `gemm_a16w16`, `gemm_a8w8` (**asm / CK / tune**), `gemm_a8w8_blockscale`, `gemm_a4w4`, **batched_gemm**, **deepgemm**, etc. |
| 3 | **Paged Attention (PA)** | `pa_fwd_naive` / **asm**, `paged_attention_v1` / **ragged**, **Gluon decode** related paths. |
| 4 | **Normalization** | `layer_norm`, `layernorm2d`, `rms_norm`, `rmsnorm2d`; often fused with **add**, **smoothquant**, **dynamicquant**, etc. |
| 5 | **Activation** | `silu_and_mul`, `scaled_silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`, `gelu_fast`. |
| 6 | **Quantization** | `per_tensor_quant`, `per_token_quant`, `smoothquant`, `dynamic_per_tensor`, **FP4** (e.g., **per_1x32_f4_quant**), **block quant**, **MXFP4**. |
| 7 | **MoE** | `fmoe`, `fmoe_g1u1`, `fmoe_fp8_blockscale_g1u1`, `moe_stage1` / `moe_stage2`, `moe_fused_gate`, `moe_sorting`, `ck_moe_stage1` / `ck_moe_stage2`. |
| 8 | **RoPE** | `rope_fwd` / `rope_bwd`, `rope_cached`, `rope_2d`. |
| 9 | **KV / Cache / MLA** | `reshape_and_cache`, `concat_and_cache_mla`, `fused_qk_rmsnorm_group_quant`, etc. |
| 10 | **Collective / Communication** | `all_reduce`, `reduce_scatter`, `all_gather`, `custom_all_reduce`, `quick_all_reduce` (some depend on the **Triton + Iris** stack). |
| 11 | **Sampling** | `greedy_sample`, `random_sample`, `topk_softmax`, `top_k_per_row`. |
| 12 | **Elementwise** | `add`, `sub`, `mul`, `div`, `sigmoid`, `tanh`, etc. |

The **feature table in the README** (MHA, MLA, PA, FusedMoe, QUANT, RMSNORM, LAYERNORM, ROPE, GEMM, etc.) is consistent with the table above and can serve as a **simplified index for external communication**.

## 3. FP8 & Architecture (gfx950 / gfx1250 vs Others)

AITER distinguishes between **OCP** and **non-OCP (FNUZ)** variants on **FP8** paths (exact type names are subject to source code):

| Target Architecture (Examples) | FP8 Orientation |
|-------------------------------|-----------------|
| **gfx950**, **gfx1250** | Uses **E4M3FN**, **E5M2**, and other **OCP**-aligned formats. |
| **Other arch** | Commonly uses **E4M3FNUZ**, **E5M2FNUZ**, and other **FNUZ** variants. |

When porting operators or comparing **MI300 vs MI355**, first confirm that the **dtype enumeration and tensor layout** are consistent with the target **ISA / OCP support** before discussing performance.

## 4. Tuning & Configuration Infrastructure

| Type | Location (Convention) | Purpose |
|------|----------------------|---------|
| **CSV** | `aiter/configs/*.csv` | **GEMM** and other parameter tables tuned **per model / scenario**. |
| **JSON** | `aiter/ops/triton/configs/**/*.json` | **JSON configurations** for **Triton kernel** block, split, etc. |
| **MoE tiering** | Logic layer commonly tiers by **token count M** | e.g., **small_M** (M < 256), **medium_M** (M < 1024), **large_M**: corresponding to different expert parallelism and kernel selection. |

## 5. Triton GEMM Configuration Parameters (Common Fields)

In JSON or Python configurations, Triton GEMM commonly features the following **tunable fields** (names subject to the specific file):

| Parameter | Function |
|-----------|----------|
| `BLOCK_SIZE_M` / `BLOCK_SIZE_N` / `BLOCK_SIZE_K` | Tile shape. |
| `GROUP_SIZE_M` | M-direction **grouping**, affects L2 locality and wave scheduling. |
| `NUM_KSPLIT` / `SPLITK_BLOCK_SIZE` | **K-dimension splitting** and **split-K** block size, used for large K or numerical strategies. |
| `cache_modifier` | **Cache hint** for global loads (e.g., `ca`, `cg`), depends on architecture and co-resident kernels. |

## 6. When to Use AITER vs Writing a Custom Kernel

| Scenario | Recommendation |
|----------|----------------|
| **Standard operator + common shapes + supported dtype** | Prefer **AITER**, and use **op_tests** plus business cases for regression. |
| **Fusions not covered by the library or special mask/layout** | **Custom kernel** (CK / Triton / asm), using the **closest AITER sub-operator** as the performance lower bound. |
| **Performance not meeting target** | First check whether the **CSV/JSON** matches the current **model tier, arch, and batch**; then consider custom work. |
| **Rare precision combinations** | First check **FP8 / INT4 / MXFP4** branches; if absent, write custom and mind the **architecture differences** section above. |

## 7. Benchmark Comparison (Illustrative)

If the repository provides benchmark scripts, use entry points consistent with the official **README / docs**; the following is a **templated** command placeholder:

```bash
# Use the actual script from the repository, for example:
# python scripts/benchmark_kernel.py --kernel my_kernel.py --op attention --baseline aiter
```

### AITER Speedups on DeepSeek-R1 / V3 Series (Public Blog Post, MI300X)

The following are approximate magnitudes relative to the baseline without **AITER** (specific models and batch sizes per the original text):

| Operator / Scenario | Speedup |
|---------------------|---------|
| **MLA decode** | **17x** |
| **Block-scale fused MoE** | **3x** |
| **Block-scale GEMM** | **2x** |
| **MHA prefill** | **14x** |
| **End-to-end (SGLang, 8xMI300X)** | **2.1x** (e.g., **6484 -> 13704 tok/s**) |

## 8. Integration with SGLang / vLLM and PD Disaggregation

### SGLang

```bash
export SGLANG_USE_AITER=1
```

Can be combined with **FlyDSL MoE**, etc. (example): `AITER_USE_FLYDSL_MOE=1` and `--disable-radix-cache --enable-torch-compile`, etc., subject to the repository and scenario.

### vLLM (ROCm)

Master switch:

```bash
export VLLM_ROCM_USE_AITER=1
```

With the master switch enabled, the following sub-switches are commonly used for fine-grained control (defaults change across versions; refer to the **vLLM** documentation):

| Environment Variable | Function |
|---------------------|----------|
| `VLLM_ROCM_USE_AITER_LINEAR` | **Linear** layer **FP8** quantized **GEMM** |
| `VLLM_ROCM_USE_AITER_MOE` | Fused **MoE** routing and computation |
| `VLLM_ROCM_USE_AITER_RMSNORM` | **RMSNorm** acceleration |
| `VLLM_ROCM_USE_AITER_MLA` | **MLA** (**DeepSeek**, etc.) |
| `VLLM_ROCM_USE_AITER_MHA` | **MHA** (**Llama/Mistral**, etc.) |
| `VLLM_ROCM_USE_AITER_FP8BMM` | **MLA** with **FP8** **batched matmul** |
| `VLLM_ROCM_USE_SKINNY_GEMM` | Small **batch** **skinny GEMM** |

**DeepSeek MLA** with **vLLM** often requires **`--block-size 1`**; otherwise it may throw errors.

### Disaggregated Serving (Prefill / Decode Separation)

**SGLang** supports **PD disaggregation**, separating **prefill** and **decode** into different processes/**GPUs**. Typical **launch_server** related parameters include:

| Parameter | Meaning |
|-----------|---------|
| `--disaggregation-mode` | `prefill` or `decode` |
| `--disaggregation-transfer-backend` | **KV** transfer backend (e.g., **mooncake**, **nixl**) |
| `--disaggregation-ib-device` | **InfiniBand** / **RoCE** device name |
| `--disaggregation-bootstrap-port` | **Bootstrap** port (crawled default **8998**) |
| `--base-gpu-id` | Starting **GPU** index on multi-card setups (commonly used on the **decode** side) |
| `--model-path` | Model path (required on both sides) |
| `--port` / `--host` | Service listening address and port |

The frontend is often paired with **`python -m sglang_router.launch_router --pd-disaggregation --prefill ... --decode ...`**. Fine-grained thread and timeout settings are in **`SGLANG_DISAGGREGATION_*`** environment variables (**thread pool**, **queue**, **heartbeat**, etc.).

## 9. DeepSeek-R1 / V3 Optimized Operators (Detailed)

The following table details the specific AITER operator backends used for DeepSeek-R1/V3 inference, as disclosed in AMD blog posts. These represent the highest-performance operator selections for this model family on MI300X.

| Component | Backend | Description |
|-----------|---------|-------------|
| **MoE Top-K Routing** | HIP kernel | Fused biased grouped top-k selection |
| **MoE Sorting** | CK | MoE alignment and sort |
| **MoE FP8 Blockscale** | Assembly | Fused FP8 blockscale group GEMM (best perf on AMD) |
| **FP8 GEMM** | CK (pre-shuffle) | Block-scale with 1x128 activation scales, 128x128 weight scales |
| **MLA Decode** | Assembly | Latent attention (head dim 576/512) with weight absorption |
| **MHA Prefill** | CK | Multi-head attention (head dim 192/128) |
| **MLA Prefill** | Assembly | Latent attention (limited to q_extend < 160) |
| **Custom AllReduce** | HIP | Optimized for MI300X inter-chip communication |

### Key API Imports

```python
from aiter import biased_grouped_topk
from aiter.fused_moe_bf16_asm import asm_moe
from aiter import gemm_a8w8_blockscale_wpreshuffle_CK
from aiter.mla import mla_decode_fwd, mla_prefill_fwd
```

### Weight Pre-shuffle for GEMM

For pre-shuffled GEMM paths (used in FP8 block-scale inference), weights must be rearranged into consumption order at model load time:

```python
from aiter.ops.shuffle import shuffle_weight
layer.weight.data = shuffle_weight(layer.weight.contiguous(), (16, 16))
```

This eliminates decode-path overhead from weight layout transformation at runtime.

### DeepSeek-R1 Performance Impact (batch=64, input=512, output=32, TP=8)

| Metric | Before AITER | After AITER | Improvement |
|--------|-------------|-------------|-------------|
| Prefill latency | 3.13s | 1.51s | **-52%** |
| Decode latency (median) | 53ms | 28ms | **-47%** |
| Total throughput | 7,332 tok/s | 14,636 tok/s | **+100%** |

---

## 10. FlyDSL MoE Integration

AITER optionally uses **FlyDSL** for mixed-precision MoE kernels (e.g., A4W4, W4A16). When FlyDSL is not installed, AITER falls back to the CK path.

### Environment Variables for FlyDSL MoE

| Variable | Purpose |
|----------|---------|
| `AITER_USE_FLYDSL_MOE=1` | Enable FlyDSL for MoE kernels |
| `AITER_USE_FLYDSL_MOE_STAGE1=1` | FlyDSL for MoE stage 1 (gate/up projection) |
| `AITER_USE_FLYDSL_MOE_STAGE2=1` | FlyDSL for MoE stage 2 (down projection) |
| `FLYDSL_W4A16_HYBRID=w2_bf16` | Mixed precision: W4A16 for stage 1, BF16 for stage 2 |

See `libraries/flydsl-reference.md` for detailed FlyDSL documentation and benchmarks.

---

## 11. Connection to CK Documentation

- **CK tile / pipeline / GemmConfig\*** details: [[ck-tile-tuning.md|ck-tile-tuning]], [[ck-programming-model.md|ck-programming-model]]
- Many GEMM / FMHA implementations in AITER still rely on **CK**; reading about **ck_tile** helps understand **tuning boundaries and bottleneck types**.
- FlashInfer as alternative attention backend: [[flashinfer-reference.md|flashinfer-reference]]
- FlyDSL for custom MoE kernels: [[flydsl-reference.md|flydsl-reference]]
