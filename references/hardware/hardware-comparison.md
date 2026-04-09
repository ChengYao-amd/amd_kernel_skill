# AMD GPU 硬件对比

## 速查表（四款 GPU）

| 参数 | MI300X (GFX942/CDNA3) | MI325X (GFX942/CDNA3) | MI355X (GFX950/CDNA4) | MI350X (GFX950/CDNA4) |
|------|------------------------|-------------------------|------------------------|------------------------|
| 计算单元 (CU) | **304**（8×38） | **304**（同左） | **256**（8×32） | **256**（同左） |
| VRAM | 192 GiB HBM3 | 256 GiB HBM3E | 288 GiB HBM3E | 288 GiB HBM3E（与 MI355X 同档规格） |
| Wavefront | 64 | 64 | 64 | 64 |
| LDS / CU | 64 KiB | 64 KiB | **160 KiB** | **160 KiB** |
| L3 Cache | 256 MiB | 256 MiB | 256 MiB | 256 MiB |
| L2 Cache | 32 MiB（4 MiB/XCD） | 32 MiB（同左） | 32 MiB（4 MiB/XCD） | 32 MiB（同左） |
| L1 Vector / CU | 32 KiB | 32 KiB | 32 KiB | 32 KiB |
| VGPR File / CU | 512 KiB | 512 KiB | 512 KiB | 512 KiB |
| SGPR File / CU | 12.5 KiB | 12.5 KiB | 12.5 KiB | 12.5 KiB |
| Matrix Core 数 | 1216（4×304） | 1216（同左） | 以 AMD 官方 CDNA4 规格为准 | 同 MI355X |
| Matrix 峰值（官方标称） | FP16 1307.4 TF；BF16 1307.4 TF；FP8 2614.9 TF；FP64 163.4 TF | 同架构，以 SKU 文档为准 | FP16 2.5 PF；FP8 5 PF；FP6 10 PF；FP4 10 PF | 同 MI355X |
| Max engine clock | 2100 MHz | 见产品规格 | 见产品规格 | 见产品规格 |
| FP8 格式侧重 | CDNA3 常见 **FNUZ** 语境 | 同左 | **OCP**（E4M3FN + E5M2） | **OCP**（同左） |
| 新特性 | — | — | FP6、FP4、MXFP | 同 MI355X |
| offload-arch | `gfx942` | `gfx942` | `gfx950` | `gfx950` |

说明：MI350X 与 MI355X 均为 **GFX950 / CDNA4**，上表核心架构参数一致；具体 SKU、功耗与认证以 AMD 官方产品页为准。

## 跨硬件移植清单

1. **编译**：更新 `--offload-arch`（`gfx942` ↔ `gfx950`），多目标示例见下。
2. **Tile/Block size**：按 CU 数、LDS/CU（64 KiB → 160 KiB 跨代时尤其重要）、内存层次重新调优。
3. **数据类型**：CDNA3 → CDNA4 迁移时核对 **FP8（FNUZ vs OCP）** 与 **FP6/FP4/MXFP** 可用性及语义。
4. **性能基线**：重新 benchmark；峰值 TFLOPS/PF 与缓存容量已随代际变化。
5. **条件编译**：使用 `#if defined(__gfx942__)` / `#if defined(__gfx950__)`（以实际编译器预定义为准）做硬件特化。
6. **Profiling 阈值**：occupancy、带宽与 MFMA 利用率「良好」区间可保留定性表，绝对数值需按设备重标定。

## 何时使用多目标编译

若代码需在 **gfx942** 与 **gfx950** 上运行：

```bash
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 kernel.cpp
```

编译器为各目标生成代码；运行时分发或条件编译再叠加各代 LDS/FP8 差异的特化路径。
