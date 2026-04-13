# General Optimization Patterns (Early and Mid-Stage)

## Memory Optimization

### 1. Coalesced Global Memory Access
- Ensure consecutive threads access consecutive addresses
- Wavefront (64 threads) should access a contiguous 256B region
- Stride-1 access pattern is ideal

### 2. Vectorized Load
- Use `float4` / `dwordx4` for 128-bit loads (4x bandwidth efficiency)
- Align data to 16-byte boundaries
- Triton: Handled automatically with correct BLOCK_SIZE settings

### 3. LDS Usage
- Use LDS for data reuse within a thread block
- Budget per CU (shared by all blocks on the CU):
  - **CDNA3** (MI300X/MI325X): **64 KB/CU**, **32 banks** x 4B, read bandwidth **128 B/clock**
  - **CDNA4** (MI350X/MI355X): **160 KB/CU**, **64 banks** x 4B, read bandwidth **256 B/clock** (2x CDNA3); supports **direct load from L1 data cache** (bypasses VGPR, reduces register pressure)
- More LDS per block -> fewer concurrent blocks -> lower occupancy
- When migrating CDNA3 -> CDNA4: bank conflict padding/swizzle formulas must be recalculated for 64 banks

## Compute Optimization

### 4. Kernel Fusion
- Fuse element-wise operations with preceding/following GEMM/reduction
- Saves global memory round trips
- Common fusions: Linear+Activation, Norm+Scale, Attention+Softmax

### 5. Tiling
- Decompose large problems into tiles that fit in LDS
- Tile size = balance between data reuse (larger) and occupancy (smaller)
- CK-specific tile recommendations in `ck-tile-tuning.md`

### 6. Loop Unrolling
- HIP uses `#pragma unroll N`, Triton handles it automatically
- Sweet spot: Enough to fill the pipeline, but not enough to cause register spilling
- Check VGPR count after unrolling

## Launch Optimization

### 7. Grid Size Setting
- **MI300X/MI325X** (CDNA3): **304 CU** (8 XCD x 38 active CU). Need at least 304 blocks for full utilization
- **MI350X/MI355X** (CDNA4): **256 CU** (8 XCD x 32 active CU). Need at least 256 blocks for full utilization
- More blocks (2-4x CU count) help hide per-block variance
- Very small kernels: Consider batching or persistent kernel

### 8. Thread Block Size
- Default: 256 (4 wavefronts per block) -- good general-purpose choice
- Minimum: 64 (1 wavefront) -- register-heavy kernels
- Maximum: 1024 (16 wavefronts) -- high occupancy demand

## Anti-Patterns (with Hardware Relevance Notes)

| Anti-Pattern | Impact | Hardware Notes |
|--------------|--------|----------------|
| Stride-N access | Bandwidth waste | All AMD GPUs |
| LDS > 32KB/block (CDNA3) | Limited to 2 blocks/CU | MI300X: **64KB/CU** (32 banks) |
| LDS > 80KB/block (CDNA4) | Limited to 2 blocks/CU | MI355X: **160KB/CU** (64 banks); larger tiles possible but re-profile occupancy |
| Using `__syncthreads` in divergent code | Deadlock risk | All AMD GPUs |
| Assuming warp=32 | Wrong reduction, wrong shuffle | AMD wavefront=64 |

---

## MIOpen: Find API and Immediate Mode

(Excerpted from the official **MIOpen** **find-and-immediate** documentation)

| Mode | API Form | Characteristics | Use Case |
|------|----------|-----------------|----------|
| **Find API** | `miopenFindConvolution*` | Compiles and **benchmarks** all **solvers**, results written to disk cache | First deployment or pursuing peak performance |
| **Immediate Mode** | `miopenConvolution*Immediate` | Queries **FindDb**, skips online **find**, fast startup | Production environments, low-latency startup |

**Find mode** (`MIOPEN_FIND_MODE`) common values: `NORMAL` (full **find**), `FAST`, `HYBRID`, **`DYNAMIC_HYBRID` (default)**, `TRUST_VERIFY`, etc.; the default strategy generally uses **FindDb** if hit, otherwise falls back to lightweight or skips some dynamic kernels.

**Immediate** fallback can optionally use **AI heuristics** (`MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK`) or weighted throughput indexing to guess a better **solver** when the cache is missing.

Typical **Immediate** flow: `GetSolutionCount` -> `GetSolution` (sorted by performance) -> optional `CompileSolution` -> `...Immediate` execution.

---

## MXFP4 / MXFP6 Quantization Workflow (AMD Quark)

**MXFP** (**OCP Microscaling**) uses **32** elements per **block** sharing an **E8M0 scale**, with elements as **FP4 (E2M1)** or **FP6 (E2M3/E3M2)**; **MI355X** and similar have native matrix paths for **FP4/FP6**, with peak approximately **4x** relative to **FP16** (depends on format/instruction, refer to whitepaper for details).

Recommended flow: **Scaling** -> **Clipping** -> **Rounding (RNE)** (omitting **RNE** can lead to noticeable precision degradation). **AMD Quark** toolchain supports **GPTQ**, **SmoothQuant**, **Quarot**, **AutoSmoothQuant**, etc., with output compatible with **vLLM** and **SGLang**; can mix **MXFP4** / **MXFP6** to balance precision and compression ratio. Public model examples are available on **Hugging Face** under **amd/**-prefixed **MXFP4** preview weights.

---

## GEAK HIP Agent Mode (Automated Kernel Optimization Case Study)

**GEAK** employs a **Generator -> Evaluator -> Reflector** loop: an **LLM** generates **HIP** rewrites, compiles and benchmarks them, and feeds failure logs back for regeneration on failure.

| Case | Agent Speedup | Comparison (Manual) |
|------|---------------|---------------------|
| **Voxelization** | **2.07x** | **1.84x** |
| **SwiGLU** | **1.68x** | **1.30x** |

**Voxelization** common patterns: **shared memory** caching of predecessor coordinates, coalesced access, **block**-level **tiling**, loop unrolling for **ILP**, `launch_bounds` hints for occupancy, early exit.

**SwiGLU** patterns: **bf16x2** pairing, **uint4** **128-bit** vectorization, **16B** alignment detection with fallback, `__expf` / `__fdividef` fast math, cross-element instruction interleaving.

The more general conclusion is: on specific kernels, **agent**-generated code can surpass experienced engineers, but still requires thorough verification and regression testing; **GEMM** size heuristics can also achieve approximately **~1.28x** relative to default without exhaustive tuning.
