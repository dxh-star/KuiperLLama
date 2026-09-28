
// ============================================================
// test_cu_emb.cpp —— CUDA Embedding 查找算子（EmbeddingKernel）单元测试
//
// Embedding 的作用：给定 token id，在词表权重表里取出对应的那一行向量。
//   输入：token id（int32）
//   权重：词表 (token_num, dim)，行号 = token id
//   输出：该 token 对应的 (dim,) 向量
//
// 这里构造一个 4x512 的"词表"，第 0 行填 0..511、第 1 行填 512..1023、
// 第 2 行填 1024..1535、第 3 行填 1536..2047，然后查 id=1 / id=2 验证取行是否正确。
// ============================================================
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

// 测试 1：默认流查 token id=1
TEST(test_emb_cu, emb1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();   // GPU 分配器
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();   // CPU 分配器

  int32_t token = 4;   // 词表有 4 个 token
  int32_t dim = 512;   // 每个 token 的向量维度
  int32_t size = 2048; // token * dim = 4 * 512，权重表总元素数

  tensor::Tensor input(base::DataType::kDataTypeFp32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 1;  // 要查的 token id = 1

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

  // 权重表：第 i 个元素填 float(i)，于是第 k 行 = [k*dim .. (k+1)*dim-1]
  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();  // 权重拷到 GPU
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token,
                                                        nullptr);  // 默认流

  output.to_cpu();  // 结果拷回 CPU
  // token id=1 对应的行 = [1*512 .. 2*512-1] = [512 .. 1023]
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 512 + i);
  }
}

// 测试 2：默认流查 token id=2（验证不同的行偏移）
TEST(test_emb_cu, emb2_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 2;  // 查 token id = 2

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token,
                                                        nullptr);

  output.to_cpu();
  // token id=2 对应的行 = [2*512 .. 3*512-1] = [1024 .. 1535]
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 1024 + i);
  }
}

// 测试 3：显式流查 token id=1（验证流参数不影响结果）
TEST(test_emb_cu, emb1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 1;

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  cudaStream_t stream;
  cudaStreamCreate(&stream);  // 显式流
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token,
                                                        stream);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 512 + i);  // 结果与默认流一致
  }

  cudaStreamDestroy(stream);
}
