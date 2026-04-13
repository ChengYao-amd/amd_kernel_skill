# FlyDSL Reference: Python-Native GPU Kernel DSL for AMD

FlyDSL is a **Python-first, MLIR-native domain-specific language** for expert GPU kernel development on AMD GPUs. It provides thread-level and IR-level control that complements Triton's block-level programming model.

**Install:** `pip install flydsl`

---

## 1. Overview and Positioning

| Aspect | Triton | FlyDSL |
|--------|--------|--------|
| Programming level | Block-level | Thread-level and IR-level |
| Target audience | Mainstream kernel developers | Expert developers targeting roofline performance |
| Layout control | Implicit | Explicit lane control, custom layouts, ISA-level hints |
| Compilation | Python -> Triton-IR -> LLVM-IR -> AMDGCN | Python DSL -> AST transforms -> Fly dialect (MLIR) -> ROCDL -> HSACO |
| Layout algebra | N/A | CuTe Layout Algebra (formal `(Shape, Stride)` representation) |

FlyDSL targets the gap between Triton (easy to write but limited control) and raw assembly (maximum performance but unmanageable). It is used in production at hyperscale on MI GPU clusters.

---

## 2. Compilation Pipeline

```
Python DSL
  -> AST transforms
  -> Fly dialect (MLIR)
    -> FLIR (Flexible Layout IR) -- layout algebra inspired by CuTe
  -> ROCDL dialect
  -> HSACO binary (gfx942 / gfx950)
```

### FLIR (Flexible Layout IR)

FLIR uses composable `(Shape, Stride)` abstractions to express:
- **Tiling**: decompose tensors into thread-block and warp tiles
- **Swizzling**: bank-conflict-free LDS access patterns
- **Vectorization**: optimal memory access widths
- Layout transformations are formal and composable, enabling portability across architectures.

---

## 3. Key Capabilities

- **Explicit lane control**: Direct control over which wavefront lanes handle which data
- **Register usage management**: Fine-grained register allocation hints
- **Custom memory layouts**: Define non-standard data layouts for LDS and global memory
- **ISA-level scheduling hints**: Insert instruction scheduling barriers, priority controls
- **JIT compilation**: Fast iteration cycle for kernel development
- **Layout-agnostic design**: Portability across GPU architectures via formal layout algebra

---

## 4. Supported Operators

| Category | Operators |
|----------|----------|
| Normalization | Softmax, LayerNorm, RMSNorm |
| Quantization | Per-tensor, per-token, block-scale |
| GEMM | Standard and fused variants |
| MoE | Fused MoE (BF16, W4A16, FP8 block-scale) |

---

## 5. Fused MoE Performance (Production Benchmark)

### Benchmark: Kimi-K2.5 MoE (tokens=16384, model_dim=7168, E=384, topk=8)

| Data Type | Triton (ms) | CK (ms) | FlyDSL (ms) | FlyDSL Speedup |
|-----------|-------------|---------|-------------|----------------|
| BF16 (A16W16) | 12.09 | gpu_fault | **8.68** | 1.39x vs Triton |
| W4A16 | 31.43 | unsupported | **9.77** | 3.22x vs Triton |

FlyDSL achieves the best performance across both precision modes, and is the only option that handles W4A16 MoE efficiently on MI300X.

### End-to-End Results: Kimi-K2.5 on MI300X (concurrency=40)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| TTFT (mean) | 33.5s | 17.7s | **-47.0%** |
| TPOT (mean) | 230ms | 71ms | **-69.2%** |
| Output throughput | 135 tok/s | 355 tok/s | **+162.4%** |
| GSM8K accuracy | 0.96 | 0.96 | Identical |

The optimization was motivated by profiling: `fused_moe` consumed 88-90% of GPU time at all concurrency levels.

---

## 6. Integration with AITER and SGLang

FlyDSL MoE kernels are integrated into the AITER operator library and can be activated in SGLang via environment variables:

```bash
# Enable FlyDSL MoE in AITER
export AITER_USE_FLYDSL_MOE=1
export AITER_USE_FLYDSL_MOE_STAGE1=1
export AITER_USE_FLYDSL_MOE_STAGE2=1

# Mixed-precision MoE: Stage1 = W4A16, Stage2 = BF16
export FLYDSL_W4A16_HYBRID=w2_bf16

# Enable AITER in SGLang
export SGLANG_USE_AITER=1
```

### Full SGLang Launch Example

```bash
AITER_USE_FLYDSL_MOE=1 \
AITER_USE_FLYDSL_MOE_STAGE1=1 \
AITER_USE_FLYDSL_MOE_STAGE2=1 \
FLYDSL_W4A16_HYBRID=w2_bf16 \
SGLANG_USE_AITER=1 \
python3 -m sglang.launch_server \
  --model /path/to/model --tp 8 \
  --disable-radix-cache \
  --enable-torch-compile \
  --disable-custom-all-reduce
```

**Note:** `--disable-radix-cache` is recommended when benchmarking with random inputs that have no shared prefixes.

---

## 7. When to Use FlyDSL

| Scenario | Recommendation |
|----------|----------------|
| Custom MoE kernel (BF16 or W4A16) | FlyDSL -- fastest development + best performance |
| Fused kernel requiring instruction-level control | FlyDSL -- explicit scheduling, lane control |
| Standard GEMM / Attention | Use AITER or CK first (more mature, better coverage) |
| Quick prototype | Use Triton (easier syntax, faster iteration) |
| Maximum absolute performance on specific kernel | FlyDSL (thread-level) or inline assembly |

---

## 8. Architecture Support

- **gfx942** (MI300X, MI325X) -- CDNA3
- **gfx950** (MI355X, MI350X) -- CDNA4

FlyDSL compiles through MLIR passes to produce optimized binaries for each target architecture.

---

## 9. Key Lessons from Production Deployments

1. **Profile first**: Identify the dominant kernel before investing in FlyDSL optimization. In Kimi-K2.5, `fused_moe` at 88-90% GPU time made it the obvious target.
2. **Mixed precision across MoE stages**: Using W4A16 for gate/up projections and BF16 for down projection balances throughput and accuracy.
3. **FlyDSL is optional in AITER**: When FlyDSL is not installed, AITER falls back to the CK path for mixed-precision MoE.
4. **MI355X memory advantage**: 288GB HBM3E (vs B200's 180GB) enables higher concurrency, amplifying throughput gains from FlyDSL-optimized kernels.

---

## Related Documentation

- AITER operator library and integration: `libraries/aiter-ops-reference.md`
- CK-Tile alternative for GEMM/Attention: `libraries/ck-tile-tuning.md`
- GEMM tuning strategies: `libraries/gemm-tuning-guide.md`
