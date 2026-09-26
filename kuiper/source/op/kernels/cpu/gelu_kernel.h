
#ifndef KUIPER_OP_KERNELS_CPU_GELU_KERNEL_H_
#define KUIPER_OP_KERNELS_CPU_GELU_KERNEL_H_
#include "tensor/tensor.h"
namespace kernel {
void gelu_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& output, void* stream);
}  // namespace kernel
#endif
