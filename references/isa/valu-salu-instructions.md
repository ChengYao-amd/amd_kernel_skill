# Vector ALU (VALU) and Scalar ALU (SALU) Reference

## VALU -- Per-Lane Computation

Operates on all 64 lanes of a wavefront, completing in 4 cycles (16 lanes per cycle).

| Category | Examples | Throughput |
|----------|----------|------------|
| FP32 arithmetic | v_add_f32, v_mul_f32, v_fma_f32 | 1 instruction per SIMD per cycle |
| FP16 Packed | v_pk_add_f16, v_pk_mul_f16 | 1 per cycle (2 FP16 per lane) |
| Type conversion | v_cvt_f32_f16, v_cvt_f16_f32 | 1 per cycle |
| Comparison | v_cmp_gt_f32 -> writes VCC | 1 per cycle |
| Bit manipulation | v_bfe_u32, v_bfi_b32 | 1 per cycle |
| Transcendental | v_rcp_f32, v_rsq_f32, v_exp_f32 | 1 per 4 cycles (shared unit) |

**Key point**: Transcendental functions are 4x slower. Avoid them in inner loops. Use polynomial approximations when possible.

## SALU -- Uniform Value Computation

Operates on a single scalar value, shared across all lanes in the wavefront.

| Category | Examples | Throughput |
|----------|----------|------------|
| Arithmetic | s_add_u32, s_mul_i32 | 1 per cycle |
| Logic | s_and_b64, s_or_b64 | 1 per cycle |
| Comparison | s_cmp_eq_u32 | 1 per cycle |
| Branch | s_cbranch_scc1 | Variable |
| Constant load | s_load_dwordx4 | ~200 cycles |

**Key point**: Move uniform computations (loop counters, addresses) to SALU, freeing VALU for data computation.

## Dual-Issue

CDNA3 can dual-issue VALU + SALU in the same cycle, provided:
- No data dependency between them
- They use different register files (VGPR vs SGPR)

Optimization: Interleave SALU address calculations with VALU data computations.
