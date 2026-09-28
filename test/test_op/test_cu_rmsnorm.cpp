
// ============================================================
// test_cu_rmsnorm.cpp —— RMSNorm 算子（CPU 与 CUDA）单元测试
//
// RMSNorm 公式：y = x / sqrt(mean(x^2) + eps) * w
// 覆盖内容：
//   1. rmsnorm_nostream   ：随机输入，CPU/CUDA 结果一致（1D）
//   2. rmsnorm_dim_stream ：按行（每行 size 个元素）做 RMSNorm 的 dim 版本，
//                           并验证"同一行多份"结果一致 + 与原 golden 一致
//   3. rmsnorm_stream     ：显式流 + 小尺寸（32）
//   4. rmsnorm_stream2    ：显式流 + 大尺寸（32*151*15）
// ============================================================
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

// 测试 1：随机 1D 输入，CPU 与 CUDA 结果一致（默认流）
TEST(test_rmsnorm_cu, rmsnorm_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);   // 输入随机
    wei_cpu.index<float>(i) = dist(mt);  // 权重随机
  }
  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);   // 拷到 GPU
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  // CUDA 端计算，然后拷回 CPU
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, nullptr);
  out_cu.to_cpu();
  // CPU 端计算作为参考
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);  // 两端一致
  }
}

// 测试 2：dim 版本（按行规约）——每个 token 一行，行内做 RMSNorm
TEST(test_rmsnorm_cu_dim, rmsnorm_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int dim_size = 4;   // 4 行
  int size = 1024;    // 每行 1024 个元素
  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, dim_size, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, dim_size, size, true, alloc_cpu);

  // 每一行的输入、权重都相同：in[j] = wei[j] = float(j)
  for (int i = 0; i < dim_size; ++i) {
    for (int j = 0; j < size; ++j) {
      wei_cpu.index<float>(j) = float(j);
      in_cpu.index<float>(i * size + j) = float(j);
    }
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // 先用 dim 版本计算每一行（注意第二个参数 1 表示按 size 个元素一组？这里调用两次：
  // 一次写入 out_cu，一次原位写回 in_cu，用于验证"多个 token 共享同一个 rmsnorm 结果"）
  kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, 1, nullptr);
  kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, in_cu, 1, nullptr);

  out_cu.to_cpu();
  in_cu.to_cpu();

  // 构造"单行"的 golden 结果：只用第 0 行的数据做一次普通 RMSNorm
  tensor::Tensor in_cpu_golden(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu_golden(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_golden(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();  // 检查 kernel 是否成功
  ASSERT_EQ(err, cudaSuccess);

  for (int j = 0; j < size; ++j) {
    wei_cpu_golden.index<float>(j) = float(j);
    in_cpu_golden.index<float>(j) = float(j);
  }
  tensor::Tensor in_cu_golden = in_cpu_golden.clone();
  tensor::Tensor wei_cu_golden = wei_cpu_golden.clone();
  tensor::Tensor out_cu_golden = out_cpu.clone();
  in_cu_golden.to_cuda();
  wei_cu_golden.to_cuda();
  out_cu_golden.to_cuda();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu_golden, wei_cu_golden,
                                                            out_cu_golden, nullptr);

  out_cu_golden.to_cpu();

  // 由于 4 行输入相同，dim 版本的每一行结果都应等于单行 golden 结果
  for (int i = 0; i < dim_size; ++i) {
    for (int j = 0; j < size; ++j) {
      ASSERT_EQ(out_cu.index<float>(i * size + j), out_cu_golden.index<float>(j))
          << "i: " << i << " j: " << j;
      ASSERT_EQ(in_cu.index<float>(i * size + j), out_cu_golden.index<float>(j))
          << "i: " << i << " j: " << j;
    }
  }
}

// 测试 3：显式流 + 小尺寸
TEST(test_rmsnorm_cu, rmsnorm_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32;

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
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}

// 测试 4：显式流 + 大尺寸
TEST(test_rmsnorm_cu, rmsnorm_stream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 15;  // 较大尺寸，验证 kernel 在大张量上的正确性

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
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}
