# System-Level Tuning for AMD Instinct GPUs

> System, OS, BIOS, and driver settings for AMD Instinct MI300X/MI325X (CDNA3) and MI350/MI355X (CDNA4) platforms. These settings are complementary to kernel-level optimization -- apply them BEFORE profiling kernels to establish a stable, high-performance baseline.

---

## 1. BIOS Settings (AMD EPYC 9004 Series)

These settings must be configured in the server BIOS for optimal GPU compute performance:

| Setting | Recommended Value | Purpose |
|---------|-------------------|---------|
| Above 4G Decoding | **Enabled** | Required for GPU large BAR support |
| SR-IOV | **Enabled** | Single-root IO virtualization |
| SMT Control | **Disable** | Better for compute-bound GPU workloads (disables hyperthreading) |
| NUMA Nodes per Socket | **Auto (NPS1)** | Unified memory domain for CPU |
| xGMI Max Speed (4-link) | **32 Gbps** | Maximum inter-chip bandwidth |
| xGMI Link Width | **Force x16** | Maximum inter-chip link width |
| IOMMU | **Enabled** | Required by ROCm |
| cTDP / Package Power Limit | **400W** | Maximum CPU power budget |
| APBDIS | **1** | Disable Data Fabric P-states for consistent latency |
| Fixed SOC P-state | **P0** | Maximum SOC frequency |
| TSME (Memory Encryption) | **Disabled** | Disable transparent memory encryption for performance |

---

## 2. GRUB / Bootloader Settings

Add to `GRUB_CMDLINE_LINUX` in `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX="pci=realloc=off iommu=pt"
```

| Parameter | Purpose |
|-----------|---------|
| `pci=realloc=off` | Unambiguous GPU BAR detection, prevents PCI resource reallocation |
| `iommu=pt` | IOMMU passthrough mode, eliminates DMA translation overhead |

After editing, run `update-grub` and reboot.

**Optional:** If GPU detection is unreliable, blacklist then manually load the driver:
```bash
# Add to GRUB: modprobe.blacklist=amdgpu
# Then after boot:
modprobe amdgpu
```

---

## 3. OS-Level Tuning

### 3.1 CPU Power Management

Disable CPU C2 idle states to prevent latency spikes from CPU wakeup:
```bash
cpupower idle-set -d 2
```

### 3.2 NUMA Auto-Balancing

Disable kernel NUMA auto-balancing for compute workloads (prevents unwanted page migration):
```bash
echo 0 > /proc/sys/kernel/numa_balancing
```

### 3.3 GPU Clock Management

Set deterministic GPU clock to reduce variance from power/clock throttling (PCC events):
```bash
# Set fixed performance level at 1900 MHz (maximum supported deterministic frequency)
rocm-smi --setperfdeterminism 1900
```

This reduces benchmark noise but may lower peak boost clock. Remove for production serving where thermal headroom matters.

### 3.4 GPU Monitoring

```bash
# Static GPU information
amd-smi static

# Real-time utilization metrics
amd-smi metric --usage

# Full system overview
rocm-smi
rocminfo    # List GPU properties and capabilities
```

---

## 4. GPU Partitioning Modes (MI300X)

MI300X supports hardware partitioning to create multiple logical GPUs from a single physical GPU. This is configured without reboot.

### 4.1 Compute Partition Modes

| Mode | XCDs per Partition | Logical GPUs per OAM | Best For |
|------|-------------------|----------------------|----------|
| **SPX** (default) | 8 | 1 | Large models, full GPU utilization |
| **DPX** | 4 | 2 | Dual model serving |
| **CPX** | 1 | 8 | Small models (<=13B), multi-tenant inference |

### 4.2 Memory Partition Modes

| Mode | NUMA Domains | Description | Compatible With |
|------|-------------|-------------|-----------------|
| **NPS1** (default) | 1 | All HBM unified | SPX, CPX |
| **NPS2** | 2 | Two memory domains | DPX |
| **NPS4** | 4 | Four memory quadrants | CPX |

### 4.3 Configuration Commands

```bash
# Set CPX partitioning (8 logical GPUs per MI300X)
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4

# Set DPX partitioning (2 logical GPUs per MI300X)
amd-smi set --gpu all --compute-partition DPX
amd-smi set --gpu all --memory-partition NPS2

# Reset to default SPX
amd-smi set --gpu all --compute-partition SPX
amd-smi set --gpu all --memory-partition NPS1

# If driver reload is needed after partition change:
sudo amd-smi reset -r
```

### 4.4 Partitioning Use Cases

**SPX + NPS1 (default):** Single large model with full 192 GB HBM and 304 CUs.

**DPX + NPS2:** Run 2 vLLM instances per physical GPU. Each instance sees ~96 GB HBM. Good for serving two different models or A/B testing.

**CPX + NPS4:** Run 8 vLLM instances per physical GPU. Best for small models (<=13B) where a single instance cannot saturate the full GPU. Also optimal for RCCL communication within a single OAM.

**CPX + NPS4 with vLLM:**
```bash
# Set partition mode
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4

# Launch vLLM on a specific logical GPU
export HIP_VISIBLE_DEVICES="$GPU_ID"
vllm serve $MODEL --port 8000
```

### 4.5 MI350 Partitioning

| Device | Compute Partitions | NPS Modes |
|--------|-------------------|-----------|
| MI355X / MI350X | CPX | NPS 2 |

---

## 5. NUMA and CPU Affinity

### 5.1 NUMA Binding for Training

Enable NUMA-aware thread binding to reduce cross-socket memory access:
```bash
export ENABLE_NUMA_BINDING=1
```

### 5.2 CPU Affinity Control

Prevent ROCm runtime from spreading threads across all CPU cores:
```bash
export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
```

This makes the ROCm runtime inherit the parent process CPU affinity rather than attempting its own thread placement.

### 5.3 Multi-Node RCCL

When running multi-node training, CPU affinity set by the job scheduler may conflict with RCCL:
```bash
export NCCL_IGNORE_CPU_AFFINITY=1
```

---

## 6. Kernel Argument Memory

### 6.1 Device Kernel Arguments

Store kernel arguments in device memory instead of host-pinned memory (default since ROCm 6.2):
```bash
export HIP_FORCE_DEV_KERNARG=1
```

Saves 2-3 microseconds per kernel launch. Critical for workloads with many small kernel dispatches.

### 6.2 Kernel Argument Pool Size

For MoE and other workloads with many concurrent kernels, increase the kernel argument pool:
```bash
export HSA_KERNARG_POOL_SIZE=12582912    # 12 MB
```

---

## 7. Scratch Memory (Register Spill)

### 7.1 Scratch Management

Prevent scratch memory reclamation between kernel launches (useful for large-register kernels):
```bash
export HSA_NO_SCRATCH_RECLAIM=1
```

### 7.2 Scratch Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `HSA_SCRATCH_SINGLE_LIMIT` | ~140 MB | Scratch threshold per XCC |
| `HSA_SCRATCH_SINGLE_LIMIT_ASYNC` | 3 GB | Async scratch threshold per XCC |

Increase these if kernels fail with scratch allocation errors.

---

## 8. RCCL Communication Tuning

### 8.1 General Multi-GPU Settings

```bash
# Increase channel count for better bandwidth (MI300X)
export NCCL_MIN_NCHANNELS=112

# Enable MSCCL++ for high-performance communication kernels
export RCCL_MSCCLPP_ENABLE=1

# Set MSCCL++ message threshold (bytes, default 1 MB)
export RCCL_MSCCLPP_THRESHOLD=1048576

# Enable fine-grained PCIe for P2P between PCIe-connected GPUs
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

### 8.2 CPX Mode RCCL Optimization

When using CPX partitioning, apply these for optimal intra-OAM allreduce:
```bash
export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=1073741824    # 1 GB
export MSCCLPP_READ_ALLRED=1
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

**Performance reference:**
- Default SPX allreduce: ~170 GB/s bus bandwidth
- Optimized CPX allreduce: ~315 GB/s (PyTorch) / ~340 GB/s (rccl-tests)

### 8.3 MSCCL++ Limitations

- Message size must be a non-zero multiple of 32 bytes.
- Does not support `hipMallocManaged` buffers.
- Allreduce only supports `float16, int32, uint32, float32, bfloat16` with sum operation.

### 8.4 RCCL Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `RCCL_MSCCL_FORCE_ENABLE=1` | Off | Force MSCCL on non-MI300X platforms (MI300X has it on by default) |
| `RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1` | Off | Allow MSCCL in multi-threaded/single-process config |
| `RCCL_MSCCLPP_ENABLE=1` | Off | Enable MSCCL++ communication kernels |
| `RCCL_MSCCLPP_THRESHOLD=<bytes>` | 1 MB | Max message size for MSCCL++ |
| `RCCL_ENABLE_CONTEXT_TRACKING=1` | Off | Enable context tracking (specific performance scenarios) |

---

## 9. Docker Configuration

### 9.1 Standard ROCm Container Launch

```bash
docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --shm-size 32G \
  --device=/dev/kfd \
  --device=/dev/dri \
  -v $MODEL_DIR:/model \
  <image_name>
```

Required flags:
- `--device=/dev/kfd --device=/dev/dri`: GPU device access
- `--ipc=host`: Shared memory for multi-GPU communication
- `--shm-size 32G`: Sufficient shared memory for RCCL/NCCL
- `--privileged`: Required for some GPU management operations

### 9.2 GPU Isolation in Containers

```bash
# Expose specific GPUs to container
export ROCR_VISIBLE_DEVICES=0,1,2,3
# or
export HIP_VISIBLE_DEVICES=0,1,2,3
```

---

## 10. Pre-Flight Validation Checklist

Run these checks after system configuration, before benchmarking:

```bash
# 1. Verify GPU detection and partition mode
rocminfo | grep "gfx"
amd-smi static

# 2. Verify clock settings
rocm-smi --showclocks

# 3. Verify NUMA balancing is disabled
cat /proc/sys/kernel/numa_balancing    # Should be 0

# 4. Verify iommu passthrough
dmesg | grep -i iommu

# 5. Verify inter-GPU bandwidth
rocm-bandwidth-test

# 6. Verify RCCL communication
# Use rccl-tests for allreduce bandwidth measurement
```

---

## 11. Performance Impact Summary

| Tuning Area | Expected Impact | Effort |
|-------------|----------------|--------|
| BIOS settings | 5-15% overall | One-time, requires reboot |
| GRUB iommu=pt | 2-5% on DMA-heavy workloads | One-time, requires reboot |
| Disable C2 states | Reduces latency variance | One-time per boot |
| Disable NUMA balancing | Prevents page migration overhead | One-time per boot |
| `HIP_FORCE_DEV_KERNARG=1` | 2-3 us per kernel launch | Environment variable |
| Deterministic clock | Reduces benchmark variance ~5-10% | Per profiling session |
| CPX partitioning | Up to 2x RCCL bandwidth | Per workload basis |
| `NCCL_MIN_NCHANNELS=112` | Better multi-GPU bandwidth | Environment variable |
