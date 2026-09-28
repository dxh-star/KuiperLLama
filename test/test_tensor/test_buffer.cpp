
// ============================================================
// test_buffer.cpp —— Buffer（内存缓冲区）单元测试
//
// Buffer 是对一段内存的封装，记录字节大小、分配器、设备类型（CPU/GPU），
// 并支持跨设备拷贝（copy_from）。本文件覆盖：
//   1. allocate        ：用 CPU 分配器申请内存，指针非空
//   2. use_external    ：包装外部已有指针（is_external = true）
//   3. cuda_memcpy1/2  ：CPU -> GPU 拷贝（再拷回 CPU 校验）
//   4. cuda_memcpy3    ：GPU -> GPU 拷贝
//   5. cuda_memcpy4    ：GPU -> CPU 拷贝
// ============================================================
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "../utils.cuh"

// 测试 1：用 CPU 分配器申请 32 字节，验证指针有效
TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  Buffer buffer(32, alloc);          // 让 Buffer 自己分配内存
  ASSERT_NE(buffer.ptr(), nullptr);  // 分配成功，指针非空
}

// 测试 2：包装外部已有指针（不拥有内存），is_external 应为 true
TEST(test_buffer, use_external) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);  // 第 3 参数=外部指针，第 4 参数=是否外部所有
  ASSERT_EQ(buffer.is_external(), true);
  delete[] ptr;
}

// 测试 3：CPU -> GPU 拷贝，再拷回 CPU 逐元素校验
TEST(test_buffer, cuda_memcpy1) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);  // 源数据 = 0,1,2,...,31
  }
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);  // CPU 侧外部 buffer
  buffer.set_device_type(DeviceType::kDeviceCPU);           // 标记为 CPU 数据
  ASSERT_EQ(buffer.is_external(), true);

  Buffer cu_buffer(size * sizeof(float), alloc_cu);  // GPU 目标 buffer
  cu_buffer.copy_from(buffer);                       // 内部按设备类型选择 CPU->GPU 拷贝

  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    // ptr[i] = float(i);
    ASSERT_EQ(ptr2[i], float(i));  // 数据被完整搬到 GPU
  }

  delete[] ptr;
  delete[] ptr2;
}

// 测试 4：与测试 3 相同流程（保留用于对照）
TEST(test_buffer, cuda_memcpy2) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);
  buffer.set_device_type(DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer.is_external(), true);

  // cpu to cuda
  Buffer cu_buffer(size * sizeof(float), alloc_cu);
  cu_buffer.copy_from(buffer);

  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], float(i));
  }

  delete[] ptr;
  delete[] ptr2;
}

// 测试 5：GPU -> GPU 拷贝（用 set_value_cu 把源 buffer 全部置为 1）
TEST(test_buffer, cuda_memcpy3) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  Buffer cu_buffer1(size * sizeof(float), alloc_cu);
  Buffer cu_buffer2(size * sizeof(float), alloc_cu);

  set_value_cu((float*)cu_buffer2.ptr(), size);  // 源 buffer 在 GPU 上全部置 1
  // cu to cu
  ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCUDA);

  cu_buffer1.copy_from(cu_buffer2);  // GPU 内拷贝

  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer1.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], 1.f);  // 目标 buffer 现在全是 1
  }
  delete[] ptr2;
}

// 测试 6：GPU -> CPU 拷贝（copy_from 后目标指针可直接在 CPU 侧读写）
TEST(test_buffer, cuda_memcpy4) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  Buffer cu_buffer1(size * sizeof(float), alloc_cu);  // GPU 源
  Buffer cu_buffer2(size * sizeof(float), alloc);     // CPU 目标
  ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCPU);

  // cu to cpu
  set_value_cu((float*)cu_buffer1.ptr(), size);  // GPU 源全部置 1
  cu_buffer2.copy_from(cu_buffer1);              // 拷回 CPU

  float* ptr2 = (float*)cu_buffer2.ptr();  // 拷完后可直接在 CPU 侧读取
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], 1.f);
  }
}
