
#pragma once
#include "tensor/tensor.h"
namespace kernel {
void layernorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& gamma,
                         const tensor::Tensor& beta, const tensor::Tensor& output, void* stream);
}  // namespace kernel
