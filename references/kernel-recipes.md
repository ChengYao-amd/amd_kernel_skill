# Kernel 参考实现骨架

常见算子的参考实现。作为起点使用，不是最终方案。

## RMSNorm (Triton)

```python
@triton.jit
def rms_norm_kernel(X, W, Y, stride, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + EPS)
    x_hat = x * rrms
    w = tl.load(W + cols, mask=mask)
    y = x_hat * w
    tl.store(Y + row * stride + cols, y, mask=mask)
```

## 融合 SwiGLU (Triton)

```python
@triton.jit
def swiglu_kernel(X, GATE, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    g = tl.load(GATE + offs, mask=mask)
    silu_g = g * tl.sigmoid(g)
    y = x * silu_g
    tl.store(Y + offs, y, mask=mask)
```

## Vector Add（HIP C++ — 最小示例）

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

## Reduction（HIP C++ — Wavefront 感知）

```cpp
__device__ float warp_reduce_sum(float val) {
    // AMD wavefront = 64，需要 6 步
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void block_reduce(const float* input, float* output, int N) {
    __shared__ float shared[16];  // 最多 16 个 wavefront/block（1024 线程）
    int tid = threadIdx.x;
    int wf_id = tid / 64;
    int lane = tid % 64;

    float val = (blockIdx.x * blockDim.x + tid < N)
                ? input[blockIdx.x * blockDim.x + tid] : 0.0f;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wf_id] = val;
    __syncthreads();

    if (wf_id == 0 && lane < (blockDim.x / 64)) {
        val = shared[lane];
        val = warp_reduce_sum(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}
```

## 说明

- 这些是起点。始终 profile 和优化。
- 根据目标硬件调整 block size 和数据类型。
- CK GEMM 配置参见 `ck-tile-tuning.md`。
- 在迭代中开发出新实现时回填本文件。
