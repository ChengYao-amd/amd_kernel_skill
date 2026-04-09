# AMD GPU 硬件对比

## 速查表（四款 GPU）

| 参数 | MI300X (GFX942/CDNA3) | MI325X (GFX942/CDNA3) | MI355X (GFX950/CDNA4) | MI350X (GFX950/CDNA4) |
|------|------------------------|-------------------------|------------------------|------------------------|
| XCD / die 上 CU | 每 XCD **40**（**38** active） | 同左 | 每 XCD **36**（**32** active） | 同左 |
| 计算单元 (CU) active | **304**（8×38） | **304**（同左） | **256**（8×32） | **256**（同左） |
| IOD 数量 | **4** | **4** | **2** | **2** |
| VRAM | 192 GiB HBM3 | 256 GiB HBM3E | 288 GiB HBM3E | 288 GiB HBM3E（与 MI355X 同档规格） |
| HBM 峰值带宽（白皮书） | 以 SKU 为准 | **6.0 TB/s** | **8 TB/s** | **8 TB/s** |
| L2 聚合读带宽 | **34.4 TB/s** | **34.4 TB/s**（同架构） | 以 CDNA4 文档为准 | 同左 |
| Infinity Cache 带宽 | **17.2 TB/s** | **17.2 TB/s**（同架构） | 以 CDNA4 文档为准 | 同左 |
| Wavefront | 64 | 64 | 64 | 64 |
| LDS / CU | 64 KiB | 64 KiB | **160 KiB** | **160 KiB** |
| L3 Cache | 256 MiB | 256 MiB | 256 MiB | 256 MiB |
| L2 Cache | 32 MiB（4 MiB/XCD） | 32 MiB（同左） | 32 MiB（4 MiB/XCD） | 32 MiB（同左） |
| L1 Vector / CU | 32 KiB | 32 KiB | 32 KiB | 32 KiB |
| VGPR File / CU | 512 KiB | 512 KiB | 512 KiB | 512 KiB |
| SGPR File / CU | 12.5 KiB | 12.5 KiB | 12.5 KiB | 12.5 KiB |
| Matrix Core 数 | 1216（4×304） | 1216（同左） | **1024**（4×256） | 同左 |
| Matrix 峰值（官方标称） | FP16 1307.4 TF；BF16 1307.4 TF；FP8 2614.9 TF；FP64 163.4 TF | 同架构，以 SKU 文档为准 | FP16 2.5 PF；FP8 5 PF；FP6 10 PF；FP4 10 PF | FP16 2.3 PF；FP8 4.6 PF；MXFP 9.2 PF（Table 2） |
| Max engine clock | 2100 MHz | 见产品规格 | **2400 MHz** | **2200 MHz** |
| TBP | 见产品规格 | 见产品规格 | **1400 W**（DLC） | **1000 W**（风冷） |
| FP8 格式侧重 | CDNA3 常见 **FNUZ** 语境 | 同左 | **OCP**（E4M3FN + E5M2） | **OCP**（同左） |
| 新特性 | — | — | FP6、FP4、MXFP | 同 MI355X |
| offload-arch | `gfx942` | `gfx942` | `gfx950` | `gfx950` |

说明：MI350X 与 MI355X 均为 **GFX950 / CDNA4**，CU/XCD、缓存与链路架构一致；**时钟、功耗、散热与 Matrix PF** 因 SKU 而异（见 MI355X 文档 **Table 2**）。

### Infinity Fabric 与 partition（代际差异摘要）

| 项目 | CDNA3（MI300X/MI325X） | CDNA4（MI350X/MI355X） |
|------|------------------------|-------------------------|
| Fabric link 速率 | **32 Gbps**/link | **38.4 Gbps**/link（约 **+20%**） |
| 每链路双向带宽 | **64 GB/s** | **76.8 GB/s** |
| P2P / aggregate（CDNA4） | — | ring **1075.2 GB/s**；aggregate **1203.2 GB/s**（白皮书） |
| Partition：QPX | **无**（常见为 SPX / DPX / CPX） | **QPX**：**2** XCD/partition，**4** partitions，约 **72 GB**/partition |
| NPS 选项 | **NPS1 / NPS2 / NPS4** 等（与平台相关） | **NPS1** 或 **NPS2**（**per-IOD**）；常推 **DPX+NPS2** |

### FLOPS/clock/CU（跨代算力密度）

下表便于核对 **per-CU** 强度；**MI355X** 上 **Matrix FP64** 相对 MI300X **减半**，**Matrix FP16/BF16/FP8/INT8（sparsity）** 及 **MXFP** 路径 **提升**（详见 `mi300x.md` / `mi355x.md` 完整表）。

| Computation | MI300X | MI355X |
|-------------|--------|--------|
| Vector FP64 / FP32 / FP16 | 128 / 256 / 256 | 同左 |
| Matrix FP64 | 256 | **128** |
| Matrix FP32 | 256 | 256 |
| Matrix FP16 / Sparsity | 2048 | **4096** |
| Matrix BF16 / Sparsity | 2048 | **4096** |
| Matrix FP8 / Sparsity | 4096 | **8192** |
| Matrix INT8 / Sparsity | 4096 | **8192** |
| Matrix MXFP6 / MXFP4 | N/A | **16384** |

**TF32**：CDNA4 **无**独立 TF32 硬件单元，由 **BF16** 等路径在软件层满足需求（白皮书）。

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
