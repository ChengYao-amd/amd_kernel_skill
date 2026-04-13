# rocWMMA API & Programming Reference

rocWMMA is a **header-only C++17 library** providing a wavefront-centric API for matrix multiply-accumulate (MMA) operations on AMD GPUs. It abstracts the underlying MFMA (CDNA) and WMMA (RDNA) instructions behind portable fragment-based operations.

**Version:** 2.2.0 | **Namespace:** `rocwmma`

---

## 1. Supported GPU Architectures

| Architecture Family | Wave Size | GPU IDs |
|---------------------|-----------|---------|
| CDNA (wave64) | 64 | gfx908, gfx90a, gfx942, gfx950 |
| RDNA (wave32) | 32 | gfx1100, gfx1101, gfx1102, gfx1200, gfx1201 |

---

## 2. Core API Functions

```cpp
// Fill a fragment with a scalar value
template<typename FragT, typename DataT>
void rocwmma::fill_fragment(FragT &frag, DataT value);

// Load matrix data from memory into a fragment (synchronous, full wavefront)
template<typename FragT, typename DataT>
void rocwmma::load_matrix_sync(FragT &frag, const DataT *data, uint32_t ldm);

// Load with explicit layout override
template<typename FragT, typename DataT>
void rocwmma::load_matrix_sync(FragT &frag, const DataT *data, uint32_t ldm, layout_t layout);

// Store fragment data to memory (synchronous)
template<typename FragT, typename DataT>
void rocwmma::store_matrix_sync(DataT *data, FragT const &frag, uint32_t ldm);

// Matrix multiply-accumulate: D = A * B + C
template<typename FragAccumOut, typename FragA, typename FragB, typename FragAccumIn>
void rocwmma::mma_sync(FragAccumOut &d, FragA const &a, FragB const &b, FragAccumIn &c);

// Workgroup synchronization (LDS fence)
void rocwmma::synchronize_workgroup();
```

### Important Synchronization Semantics

- `load_matrix_sync` / `store_matrix_sync` synchronize for **global memory** only.
- For **LDS** (shared memory), you must explicitly call `synchronize_workgroup()`.

---

## 3. Fragment Class Template

```cpp
template<typename MatrixT,      // matrix_a | matrix_b | accumulator
         uint32_t FragM,        // fragment M dimension
         uint32_t FragN,        // fragment N dimension
         uint32_t FragK,        // fragment K dimension
         typename DataT,        // data type (f16, bf16, f32, i8, f8, f64, etc.)
         typename DataLayoutT,  // row_major | col_major
         typename Scheduler>    // scheduling strategy (see Section 5)
class rocwmma::fragment;
```

### Key Fragment Methods

| Method | Description |
|--------|-------------|
| `height()` / `width()` | Geometric dimensions of the fragment |
| `blockDim()` / `kDim()` | Block and K dimensions |
| `size()` | Number of unpacked elements |
| `operator[]` | Per-element access |
| `operator*()` | Packed storage access |

### Critical Constraint

Fragment storage uses packed registers internally. **Element ordering and spatial locality within a fragment are not guaranteed.** Do not assume any particular memory layout when accessing fragment elements directly.

---

## 4. Supported Data Type Combinations

| Input (A/B) | Output (C/D) | Compute | BlockM | BlockN | BlockK | Architecture |
|-------------|-------------|---------|--------|--------|--------|--------------|
| f16 | f32 | f32 | 16 | 16 | 16 | All gfx9 |
| f16 | f32 | f32 | 32 | 32 | 8 | All gfx9 |
| bf16 | f32 | f32 | 16 | 16 | 16 | gfx90a+ |
| bf16 | f32 | f32 | 32 | 32 | 8 | gfx90a+ |
| i8 | i32 | i32 | 16 | 16 | 16-64 | gfx908+ |
| f8 | f32 | f32 | 16 | 16 | 32+ | gfx940+ |
| f64 | f64 | f64 | 16 | 16 | 4+ | gfx90a+ |

BlockM/N values above are **minimum recommended**. Values below these minimums use padding internally, which degrades performance.

---

## 5. Fragment Scheduling Strategies

Scheduling strategies control how waves within a thread block cooperate on fragment operations.

| Scheduler | Description |
|-----------|-------------|
| `default_schedule` | Each wave operates independently |
| `coop_row_major_2d` | Waves cooperate in 2D thread block, row-major order |
| `coop_col_major_2d` | Waves cooperate in 2D thread block, column-major order |
| `coop_row_slice_2d` | Waves in the same row cooperate |
| `coop_col_slice_2d` | Waves in the same column cooperate |
| `single` | Only the designated wave participates |

---

## 6. Thread Block Size Constraints

rocWMMA supports up to **4 wavefronts per thread block**. Valid configurations:

| TBlock_X | TBlock_Y | Total Waves |
|----------|----------|-------------|
| WaveSize | 1 | 1 |
| WaveSize | 2 | 2 |
| WaveSize | 4 | 4 |
| WaveSize * 2 | 1 | 2 |
| WaveSize * 2 | 2 | 4 |
| WaveSize * 4 | 1 | 4 |

Where `WaveSize` = 64 (CDNA) or 32 (RDNA).

---

## 7. Three-Layer Implementation Architecture

| Layer | Function |
|-------|----------|
| **Unit Backend** | Wraps `amdgcn_*` intrinsics; handles per-architecture differences |
| **Vector Operations** | Handles variable-length vectors; unrolls into unit backend calls |
| **Fragment Operations** | Wavefront-level API; translates fragment ops into vector ops |

---

## 8. GEMM Kernel Naming Convention

rocWMMA GEMM kernels follow a structured naming scheme:

| Prefix | Meaning |
|--------|---------|
| PGR0 / PGR1 | Global Read Prefetch: 0 = none, 1 = 1 stage |
| LB0 / LB2 | LDS Buffer: 0 = no LDS, 2 = double-buffered |
| MP0 / MP1 | MFMA Priority: 0 = default, 1 = elevated |
| SB / MB | Single / Multiple output blocks per wave |
| NC / CP | Non-Cooperative / Cooperative load/store |
| BLK / WV / WG | Cooperation granularity: block tile / wave tile / macro tile |

---

## 9. hipRTC Compatibility

rocWMMA is compatible with HIP Runtime Compilation (hipRTC), enabling use in dynamically compiled kernels at runtime.

---

## 10. Programming Model Guidelines

1. **Wavefront-centric**: All load/store/MMA functions assume the **entire wavefront participates**. If any threads in the wavefront are inactive, behavior is **undefined**.
2. **Larger fragments improve bandwidth utilization**: Prefer larger fragment sizes when register pressure allows.
3. **Partial fragments**: rocWMMA 2.0.0+ handles partial and oversized tiles automatically in `mma_sync`, internally padding to supported BlockMNK shapes.
4. **Compile-time verification**: Fragment dimensions and types are validated at compile time via template constraints.

---

## Related Documentation

- MFMA instruction details and register layouts: `isa/mfma-instructions.md`
- HIP compiler intrinsics for direct MFMA usage: `libraries/hip-intrinsics.md`
- CK-Tile GEMM pipeline (alternative high-performance GEMM path): `libraries/ck-tile-tuning.md`
