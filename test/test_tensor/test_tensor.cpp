
// ============================================================
// test_tensor.cpp —— Tensor（张量）单元测试
//
// Tensor = 形状(dims) + 数据类型 + Buffer（底层内存）。
// 本文件覆盖：
//   1. to_cpu      ：GPU 张量拷回 CPU，验证设备类型与数据
//   2. clone_cuda  ：克隆 GPU 张量，验证克隆体独立且数据一致
//   3. clone_cpu   ：克隆 CPU 张量
//   4. to_cu       ：CPU 张量拷到 GPU
//   5. init1       ：用分配器创建（allocate=true），非空
//   6. init2       ：不分配内存（allocate=false），为空
//   7. init3       ：用外部指针构造，不拷贝，指针与数据一致
//   8. assign1     ：把外部 Buffer 赋给张量
// ============================================================
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "../utils.cuh"
#include "base/buffer.h"

// 测试 1：GPU 张量 to_cpu() 后应变成 CPU 设备、数据可读
TEST(test_tensor, to_cpu) {
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);  // 32x32 fp32 on GPU
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32);  // GPU 上全部置 1

  t1_cu.to_cpu();  // 拷贝到 CPU 并更新设备类型
  ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);
  float* cpu_ptr = t1_cu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(cpu_ptr + i), 1.f);  // 数据完整搬过来
  }
}

// 测试 2：克隆 GPU 张量（克隆体应有自己的内存，且内容相同）
TEST(test_tensor, clone_cuda) {
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);

  tensor::Tensor t2_cu = t1_cu.clone();  // 深拷贝
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);  // 克隆体内容正确
  }

  // 原张量内容也仍是 1（克隆未破坏源数据）
  cudaMemcpy(p2, t1_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  ASSERT_EQ(t2_cu.data_type(), base::DataType::kDataTypeFp32);  // 数据类型保持
  ASSERT_EQ(t2_cu.size(), 32 * 32);                             // 元素个数保持

  t2_cu.to_cpu();  // 拷回 CPU 再校验一次
  std::memcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

// 测试 3：克隆 CPU 张量
TEST(test_tensor, clone_cpu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  for (int i = 0; i < 32 * 32; ++i) {
    t1_cpu.index<float>(i) = 1.f;  // 全部置 1
  }

  tensor::Tensor t2_cpu = t1_cpu.clone();
  float* p2 = new float[32 * 32];
  std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);  // 克隆体内容正确
  }

  std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);  // 源张量不受影响
  }
  delete[] p2;
}

// 测试 4：CPU 张量 to_cuda() 后数据在 GPU 上可读
TEST(test_tensor, to_cu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  float* p1 = t1_cpu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    *(p1 + i) = 1.f;  // CPU 侧写满 1
  }

  t1_cpu.to_cuda();  // 拷到 GPU（指针与设备类型随之更新）
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(p2 + i), 1.f);  // GPU 上数据正确
  }
  delete[] p2;
}

// 测试 5：allocate=true，张量非空
TEST(test_tensor, init1) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);  // true=分配内存
  ASSERT_EQ(t1.is_empty(), false);
}

// 测试 6：allocate=false，张量为空（只记形状，不申请内存）
TEST(test_tensor, init2) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, false, alloc_cu);  // false=不分配
  ASSERT_EQ(t1.is_empty(), true);
}

// 测试 7：用外部指针构造，不拷贝，指针与首个元素都应与外部一致
TEST(test_tensor, init3) {
  using namespace base;
  float* ptr = new float[32];
  ptr[0] = 31;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);  // 挂外部指针
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.ptr<float>(), ptr);      // 直接用外部内存，无拷贝
  ASSERT_EQ(*t1.ptr<float>(), 31);      // 数据一致
}

// 测试 8：assign(buffer) 把外部 Buffer 赋给张量
TEST(test_tensor, assign1) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);

  int32_t size = 32 * 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);  // 外部数据 0..1023
  }
  std::shared_ptr<Buffer> buffer =
      std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
  buffer->set_device_type(DeviceType::kDeviceCPU);

  ASSERT_EQ(t1_cpu.assign(buffer), true);   // 赋值成功
  ASSERT_EQ(t1_cpu.is_empty(), false);
  ASSERT_NE(t1_cpu.ptr<float>(), nullptr);  // 指向了新 buffer
  delete[] ptr;
}
