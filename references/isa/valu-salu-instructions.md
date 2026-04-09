# 向量 ALU (VALU) 与标量 ALU (SALU) 参考

## VALU — 每 Lane 计算

对 wavefront 的全部 64 个 lane 操作，4 个周期完成（每周期 16 lane）。

| 类别 | 示例 | 吞吐量 |
|------|------|--------|
| FP32 算术 | v_add_f32, v_mul_f32, v_fma_f32 | 每 SIMD 每周期 1 条 |
| FP16 Packed | v_pk_add_f16, v_pk_mul_f16 | 每周期 1 条（每 lane 2 个 FP16） |
| 类型转换 | v_cvt_f32_f16, v_cvt_f16_f32 | 每周期 1 条 |
| 比较 | v_cmp_gt_f32 → 写入 VCC | 每周期 1 条 |
| 位操作 | v_bfe_u32, v_bfi_b32 | 每周期 1 条 |
| 超越函数 | v_rcp_f32, v_rsq_f32, v_exp_f32 | 每 4 周期 1 条（共享单元） |

**关键**：超越函数慢 4 倍。避免在内层循环使用。可能时用多项式近似替代。

## SALU — 统一值计算

对单个标量值操作，wavefront 内所有 lane 共享。

| 类别 | 示例 | 吞吐量 |
|------|------|--------|
| 算术 | s_add_u32, s_mul_i32 | 每周期 1 条 |
| 逻辑 | s_and_b64, s_or_b64 | 每周期 1 条 |
| 比较 | s_cmp_eq_u32 | 每周期 1 条 |
| 分支 | s_cbranch_scc1 | 可变 |
| 常量加载 | s_load_dwordx4 | ~200 周期 |

**关键**：将统一计算（循环计数器、地址）移到 SALU，释放 VALU 给数据计算。

## 双发射

CDNA3 可在同一周期双发射 VALU + SALU，条件：
- 两者之间无数据依赖
- 使用不同寄存器文件（VGPR vs SGPR）

优化：将 SALU 地址计算与 VALU 数据计算交错排列。
