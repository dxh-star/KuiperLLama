
// ============================================================
// test_cu_add.cpp —— CUDA 向量加法算子（AddKernel）单元测试
//
// 覆盖内容：
//   1. add1_nostream ：在默认 CUDA 流（stream=nullptr）上执行 32*151 个元素的逐元素加法
//   2. add1_stream   ：在自定义 CUDA 流上执行同样的加法
//   3. add_align1    ：更大的张量（32*151*13），验证 kernel 能处理非对齐边界（尾块）
//
// 测试方法：t1 填 2.0、t2 填 3.0，调用 get_add_kernel(CUDA)，期望逐元素结果为 5.0
// （或 2.1+3.3=5.4，用 ASSERT_NEAR 容忍浮点误差）。
// ============================================================
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

// 测试 1：默认流上的向量加法
TEST(test_add_cu, add1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();  // CUDA 分配器

  int32_t size = 32 * 151;  // 元素个数（故意不取 2 的整数次幂，验证 kernel 的尾块处理）

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  // 用工具函数在 GPU 上把 t1 全部置为 2.f、t2 全部置为 3.f
  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  // stream 传 nullptr => 使用默认流
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();  // 等待 kernel 完成
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);  // 拷回 CPU
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);  // 2 + 3 = 5，逐元素精确相等
  }

  delete[] output;
}

// 测试 2：自定义流上的向量加法
TEST(test_add_cu, add1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  cudaStream_t stream;
  cudaStreamCreate(&stream);  // 创建显式流，验证 kernel 在非默认流上也能正确执行
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }
  cudaStreamDestroy(stream);
  delete[] output;
}

// 测试 3：更大张量 + 浮点加法（验证非对齐边界与浮点精度）
TEST(test_add_cu, add_align1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 13;  // 更大的元素个数

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)( t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f, 0.1f);  // 2.1+3.3=5.4，容忍浮点误差
  }

  delete[] output;
}
