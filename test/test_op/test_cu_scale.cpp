
// ============================================================
// test_cu_scale.cpp —— Scale 算子（逐元素乘以标量）测试
//
// 注意：本文件的用例目前被整体注释掉（未启用）。
// 它们验证 kernel::get_scale_kernel(CUDA)：
//   scale1_nostream：默认流，0.5f * 2.0f 期望得到 1.0f
//   scale1_stream  ：显式流，0.4f * 2.0f 期望得到 0.8f
// 需要恢复时去掉注释即可。
// ============================================================
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
// TEST(test_scale_cu, scale1_nostream) {
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   int32_t size = 32 * 151;
//
//   tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
//   set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
//   kernel::get_scale_kernel(base::DeviceType::kDeviceCUDA)(0.5f, t1, nullptr);
//   cudaDeviceSynchronize();
//
//   t1.to_cpu();
//   for (int i = 0; i < size; ++i) {
//     ASSERT_EQ(t1.index<float>(i), 1.f);
//   }
// }
//
// TEST(test_scale_cu, scale1_stream) {
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   int32_t size = 32 * 151;
//
//   tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
//   set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);
//   kernel::get_scale_kernel(base::DeviceType::kDeviceCUDA)(0.4f, t1, nullptr);
//   cudaDeviceSynchronize();
//
//   t1.to_cpu();
//   cudaStreamDestroy(stream);
//
//   for (int i = 0; i < size; ++i) {
//     ASSERT_EQ(t1.index<float>(i), 0.8f);
//   }
// }
