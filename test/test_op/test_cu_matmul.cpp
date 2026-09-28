
// ============================================================
// test_cu_matmul.cpp —— MatMul 算子（CPU 与 CUDA 双后端）单元测试
//
// MatMul 约定：y = W @ x（W 为 (out,in)，x 为 (in,)，y 为 (out,)）。
// 覆盖内容：
//   1. matmul_linear_stream5 ：CPU 与 CUDA 结果逐元素一致（4x4 权重）
//   2. matmul_linear_course  ：手算的小例子，验证 CPU 端计算正确
//   3. matmul_linear_course_cuda：同一个手算例子在 CUDA 端验证
// ============================================================
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
using namespace kernel;

// 测试 1：CPU 与 CUDA 结果一致（4 输入 -> 4 输出，权重 4x4）
TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);   // x (4,)
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu); // W (4,4)

  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = float(i);  // x = [0,1,2,3]
  }
  for (int i = 0; i < 16; ++i) {
    weight.index<float>(i) = float(i); // W = [0..15] 行主序
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);   // x 拷到 GPU
  weight.to_cuda(nullptr);  // W 拷到 GPU

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 4, true, alloc_cu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;  // 显式流
  // CUDA 端 matmul（scale=1.f）
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, config);
  // CPU 端 matmul 作为参考
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          config);

  out_cu.to_cpu();  // 拷回 CPU 比较
  for (int i = 0; i < out_cu.size(); ++i) {
    ASSERT_EQ(out_cu.index<float>(i), out_cpu.index<float>(i));  // 两端逐元素一致
  }
}

// 测试 2：手算例子，验证 CPU matmul 数学正确性
//   W = [[1..3],[4..6],[7..9]] 行主序，x = [1,1,-1]
//   y[0] = 1*1+2*1+3*(-1) = 0
//   y[1] = 4*1+5*1+6*(-1) = 3
//   y[2] = 7*1+8*1+9*(-1) = 6
TEST(test_matmul_cu, matmul_linear_course) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);  // W = 1..9
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          nullptr);

  // 手算验证
  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

// 测试 3：同一个手算例子在 CUDA 端验证
TEST(test_matmul_cu, matmul_linear_course_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 3, true, alloc_cu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_cu.clone();
  out_cpu.to_cpu();  // 结果拷回 CPU

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}
