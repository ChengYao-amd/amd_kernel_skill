# rocprof / rocprofv3 Usage Guide

The primary tool for **kernel statistics, hardware counters, and timeline traces** on ROCm has evolved to **`rocprofv3`** provided by the **ROCprofiler SDK** (typically installed at `/opt/rocm/bin/rocprofv3`). The legacy `rocprof` (rocprof v1) still coexists in some distributions for backward compatibility with old scripts; new projects should prioritize **rocprofv3** as per the official documentation. See [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).

## Relationship with ROCm Systems Profiler (rocprofiler-systems)

**ROCm Systems Profiler** (package/command commonly written as **`rocprofiler-systems`**, formerly known by names including Omnitrace) focuses on **application-level and system-level** collection: CPU/GPU coordination, sampling, optional dynamic instrumentation, system metrics, etc., and can use **Perfetto** for interactive trace browsing. It complements **`rocprofv3`**: the latter is closer to **GPU kernel, HIP/HSA API, PC sampling, hardware counter** and other low-overhead analysis. For system-level bottlenecks (threads, host side, multi-process), prefer **rocprofiler-systems**; for kernel and counter details, prefer **rocprofv3**. Documentation entry point: [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/).

> Tip: Some ROCm versions have compatibility notes regarding Perfetto UI versions; if traces display abnormally, check the current ROCm release notes for Perfetto version recommendations.

## rocprofv3: Basic Capabilities (concepts)

Typical capabilities of **`rocprofv3`** as described in official documentation include (refer to current ROCm documentation):

- Multiple **output formats**: such as **CSV, JSON, PFTrace, OTF2, rocpd (SQLite)**, etc., for consumption by scripts and visualization tools.
- **Runtime trace**: such as HIP runtime, kernel dispatch, memory activity, markers, etc. (specific options per `--help`).
- **PC sampling**, **hardware counters**, and other architecture-dependent collection capabilities.
- Support for attaching to running processes by **PID** and other methods (see official *Using rocprofv3*).

The following are **illustrative** command forms; **actual flag names should be verified against local `rocprofv3 --help` and current ROCm documentation**.

```bash
# View available options and output formats
rocprofv3 --help

# Example: trace / counter collection for an application (placeholder: replace with actual subcommands and parameters from documentation)
rocprofv3 <trace-or-counter-options> -- python run_kernel.py
```

## Timeline Trace and Perfetto UI

**rocprofv3** can output **PFTrace** (Perfetto protocol buffer) and other formats for viewing **timelines** in **Perfetto UI** ([https://ui.perfetto.dev](https://ui.perfetto.dev)): kernel launch order, overlap, correspondence with APIs, etc.

Recommended workflow:

1. Use **rocprofv3** to generate **PFTrace** (or an intermediate format documented as convertible to PFTrace, e.g., via `rocpd` conversion).
2. Open **Perfetto UI** in a browser and load the generated trace file.
3. Analyze overlap and idle periods using **kernel name**, **timestamp**, **queue**, and other tracks.

If the default output is **rocpd (SQLite)**, follow [Using rocpd output format](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocpd-output-format.html) to convert the database to **pftrace** / **otf2** etc. before analyzing with Perfetto or third-party tools.

## MI300X (MI300 Series / gfx942) Counter Examples

Counters and metrics available for MI300 and MI200 series are systematically documented in the official docs: [MI300 and MI200 series performance counters and metrics](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html). The following names can be used with **rocprof** family tools' **PMC input files** or **rocprofv3** counter lists (**exact syntax per current tool `--list-counters` / documentation**).

**Shader / Instruction Mix (SQ) -- suitable for determining compute vs MFMA vs memory access:**

| Counter (example) | Summary |
|--------------------|---------|
| `SQ_INSTS` | Total instructions issued |
| `SQ_INSTS_VALU` | VALU instructions (whether MFMA is included depends on documentation) |
| `SQ_INSTS_MFMA` | Matrix FMA (MFMA) instruction count |
| `SQ_INSTS_VALU_MFMA_F16` / `SQ_INSTS_VALU_MFMA_BF16` / `SQ_INSTS_VALU_MFMA_F32` / `SQ_INSTS_VALU_MFMA_F64` / `SQ_INSTS_VALU_MFMA_I8` | MFMA issue classification by precision |
| `SQ_INSTS_VMEM_RD` / `SQ_INSTS_VMEM_WR` | Vector memory read/write instructions |
| `SQ_INSTS_LDS` | LDS instructions (on MI300, whether flat is counted differs from MI200; see official table notes) |
| `SQ_WAVES` | Wavefronts dispatched to sequencer (includes restores etc.; see documentation for definition) |
| `SQ_VALU_MFMA_BUSY_CYCLES` | Matrix FMA ALU busy cycles |

**MFMA Throughput (MOPS unit, commonly aligned with 512 in documentation):**

| Counter (example) | Summary |
|--------------------|---------|
| `SQ_INSTS_VALU_MFMA_MOPS_F16` | F16 MFMA operation volume (unit as defined in documentation) |
| `SQ_INSTS_VALU_MFMA_MOPS_BF16` | BF16 MFMA operation volume |
| `SQ_INSTS_VALU_MFMA_MOPS_F32` / `F64` / `I8` | Other precision MFMA operation volumes |

**L2 / Texture Path (excerpt, MI300 multi-instance with `[n]` suffix):**

| Counter (example) | Summary |
|--------------------|---------|
| `TCC_HIT` / `TCC_MISS` (and `_sum` aggregate form, depending on tool version) | L2 hit/miss |
| `TA_FLAT_READ_WAVEFRONTS[n]` | Wavefronts processed by TA on flat read path |
| `TA_BUFFER_READ_WAVEFRONTS[n]` | Buffer read wavefront count |

**LDS Contention:**

| Counter (example) | Summary |
|--------------------|---------|
| `SQ_LDS_BANK_CONFLICT` | Stall cycles due to LDS bank conflicts |

When writing PMC files, the legacy **`rocprof -i counters.txt`** style is still common in tutorials; **rocprofv3** may use **XML/YAML/CLI** and other new configuration methods -- please refer to [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) for migration.

## Output Interpretation (same principles as legacy rocprof)

| Counter/Metric | Meaning | Action Direction |
|----------------|---------|------------------|
| `SQ_WAVES` | Wavefront issue count | Is grid/block too small, occupancy |
| `SQ_INSTS_VALU` | VALU activity | High may indicate compute-heavy or VALU-intensive |
| `SQ_INSTS_MFMA` / MFMA MOPS | Matrix core usage | GEMM-class kernels should have significantly non-zero values |
| `SQ_INSTS_LDS` / `SQ_LDS_BANK_CONFLICT` | LDS usage and conflicts | Check bank conflicts, shared memory patterns |
| L2 `HIT`/`MISS` | Cache behavior | Low hit rate -> optimize reuse and access patterns |

## Recommended Workflow

1. Use **rocprofv3** (or **rocprofiler-systems**, depending on bottleneck level) for the first round of **timeline / hot kernel** identification.
2. For hot kernels, configure **hardware counters** to distinguish **MFMA / VALU / VMEM / LDS / L2**.
3. When **cross CPU-GPU** or **system resource** views are needed, overlay **rocprofiler-systems** with **Perfetto**.
4. After optimization, repeat measurements; on MI300X, pay attention to **multi-GCD** and **dispatch distribution**.

## Reference Links

- [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [MI300 and MI200 performance counters and metrics](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html)
- [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)
