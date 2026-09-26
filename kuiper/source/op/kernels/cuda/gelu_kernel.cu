
#include <device_launch_parameters.h>
#include "gelu_kernel.cuh"
namespace kernel {
__global__ void gelu_f32(const float* in, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float x = in[i];
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    out[i] = 0.5f * x * (1.0f + tanhf(inner));
  }
}
void gelu_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty()); CHECK(!output.is_empty());
  CHECK(input.device_type()==base::DeviceType::kDeviceCUDA && output.device_type()==base::DeviceType::kDeviceCUDA);
  const int32_t size=static_cast<int32_t>(input.size());
  const float* in_ptr=input.ptr<float>();
  float* out_ptr=const_cast<float*>(output.ptr<float>());
  constexpr int threads_num=128;
  const int32_t blocks=(size+threads_num-1)/threads_num;
  if (stream) {
    cudaStream_t s=static_cast<cudaStream_t>(stream);
    gelu_f32<<<blocks,threads_num,0,s>>>(in_ptr,out_ptr,size);
  } else {
    gelu_f32<<<blocks,threads_num>>>(in_ptr,out_ptr,size);
  }
}
}  // namespace kernel
