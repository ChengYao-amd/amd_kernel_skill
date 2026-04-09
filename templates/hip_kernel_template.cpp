/*
 * HIP C++ kernel template for AMD GPUs.
 * Compile: hipcc -O3 --offload-arch=gfx942 -shared -fPIC -o kernel.so kernel.cpp \
 *          $(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch/TorchConfig.cmake
 *
 * For multi-target: --offload-arch=gfx942 --offload-arch=gfx950
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>

// === Kernel ===

template <typename T, int BLOCK_SIZE = 256>
__global__ void kernel_op_fwd(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int N
) {
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= N) return;

    // Wavefront = 64 on AMD (not 32)
    // const int lane_id = threadIdx.x % 64;
    // const int wf_id = threadIdx.x / 64;

    T val = input[idx];

    // === Kernel logic here ===

    output[idx] = val;
}

// === PyTorch binding ===

torch::Tensor kernel_op(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int N = input.numel();
    constexpr int BLOCK_SIZE = 256;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "kernel_op", [&] {
            kernel_op_fwd<scalar_t, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N
            );
        }
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_op", &kernel_op, "Custom HIP kernel");
}
