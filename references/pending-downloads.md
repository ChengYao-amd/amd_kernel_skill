# 待下载 PDF 文档清单

下载到 `/apps/tas/0_public/agent_refs/pdfs/` 目录。

## ISA 规格文档（最高优先级）

| 文件名 | 链接 | 说明 |
|--------|------|------|
| `cdna3-isa.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf | CDNA3 ISA 完整指令集参考（MI300X/MI325X） |
| `cdna4-isa.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf | CDNA4 ISA 完整指令集参考（MI355X/MI350X） |

## 架构白皮书

| 文件名 | 链接 | 说明 |
|--------|------|------|
| `cdna3-whitepaper.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf | CDNA3 架构白皮书 |
| `cdna4-whitepaper.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf | CDNA4 架构白皮书 |

## 产品规格手册

| 文件名 | 链接 | 说明 |
|--------|------|------|
| `mi355x-brochure.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf | MI355X GPU 产品手册 |
| `mi355x-platform-brochure.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-miI355x-platform-brochure.pdf | MI355X 平台手册 |
| `mi300x-datasheet.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf | MI300X 平台数据表 |

## 旧架构参考（对比用）

| 文件名 | 链接 | 说明 |
|--------|------|------|
| `cdna2-isa.pdf` | https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf | CDNA2 ISA（MI200 系列） |
| `cdna2-whitepaper.pdf` | https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf | CDNA2 架构白皮书 |

## 下载命令

```bash
cd /apps/tas/0_public/agent_refs/pdfs

# ISA 文档
wget -O cdna3-isa.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf"
wget -O cdna4-isa.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf"

# 白皮书
wget -O cdna3-whitepaper.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf"
wget -O cdna4-whitepaper.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf"

# 产品手册
wget -O mi355x-brochure.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf"
wget -O mi355x-platform-brochure.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-miI355x-platform-brochure.pdf"
wget -O mi300x-datasheet.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf"

# 旧架构参考
wget -O cdna2-isa.pdf "https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf"
wget -O cdna2-whitepaper.pdf "https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf"
```

## 下载后阅读优先级

1. **cdna4-isa.pdf** → 补充 `references/isa/mfma-instructions.md`（CDNA4 完整 MFMA 指令 + 寄存器布局）
2. **cdna4-whitepaper.pdf** → 补充 `references/hardware/mi355x.md`（微架构细节、cache hierarchy、Infinity Fabric）
3. **cdna3-isa.pdf** → 校验 `references/isa/` 全部文档中的指令延迟、编码格式
4. **mi355x-brochure.pdf** → 确认 MI355X 峰值性能数字
5. **cdna3-whitepaper.pdf** → 补充 MI300X XCD 内部结构、DME 细节
