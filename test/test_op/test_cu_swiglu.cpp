
// ============================================================
// test_cu_swiglu.cpp —— SwiGLU 激活算子（CPU 与 CUDA）单元测试
//
// SwiGLU：y = silu(x1) * x2，其中 silu(x) = x * sigmoid(x)
// 覆盖内容：
//   1. swiglu_nostream ：默认流，随机输入，CPU/CUDA 结果一致
//   2. swiglu_stream   ：显式流，同样的随机输入对比
// ============================================================
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

// 测试 1：默认流，CPU/CUDA 结果一致
TEST(test_swiglu_cu, swiglu_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  // 两个输入张量（gate 与 up）
  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  // CUDA 端计算并拷回
  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu,
                                                           nullptr);
  out_cu.to_cpu();

  // CPU 端作为参考
  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                          nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);  // 两端一致
  }
}

// 测试 2：显式流，同样的对比
TEST(test_swiglu_cu, swiglu_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);  // 显式流

  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                          nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}
