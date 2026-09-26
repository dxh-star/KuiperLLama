
#include "gelu_kernel.h"
#include <cmath>
namespace kernel {
void gelu_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty()); CHECK(!output.is_empty());
  CHECK(input.device_type()==base::DeviceType::kDeviceCPU && output.device_type()==base::DeviceType::kDeviceCPU);
  const float* in_ptr=input.ptr<float>();
  float* out_ptr=const_cast<float*>(output.ptr<float>());
  const int32_t size=static_cast<int32_t>(input.size());
  const float sqrt_2_over_pi=0.7978845608028654f;
  const float coeff=0.044715f;
  for(int32_t i=0;i<size;++i){
    float x=in_ptr[i];
    float inner=sqrt_2_over_pi*(x+coeff*x*x*x);
    out_ptr[i]=0.5f*x*(1.0f+std::tanh(inner));
  }
}
}  // namespace kernel
