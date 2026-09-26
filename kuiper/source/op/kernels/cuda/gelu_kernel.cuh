
#pragma once
#include "tensor/tensor.h"
namespace kernel {
void gelu_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& output, void* stream);
}  // namespace kernel
