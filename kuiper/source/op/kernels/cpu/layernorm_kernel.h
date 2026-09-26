
#ifndef KUIPER_OP_KERNELS_CPU_LAYERNORM_KERNEL_H_
#define KUIPER_OP_KERNELS_CPU_LAYERNORM_KERNEL_H_
#include "tensor/tensor.h"
namespace kernel {
void layernorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& gamma,
                          const tensor::Tensor& beta, const tensor::Tensor& output, void* stream);
}  // namespace kernel
#endif
