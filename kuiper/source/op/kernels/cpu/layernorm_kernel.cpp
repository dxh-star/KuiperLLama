
#include "layernorm_kernel.h"
namespace kernel {
void layernorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& gamma,
                          const tensor::Tensor& beta, const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty()); CHECK(!gamma.is_empty()); CHECK(!beta.is_empty()); CHECK(!output.is_empty());
  CHECK(input.device_type()==base::DeviceType::kDeviceCPU && gamma.device_type()==base::DeviceType::kDeviceCPU &&
        beta.device_type()==base::DeviceType::kDeviceCPU && output.device_type()==base::DeviceType::kDeviceCPU);
  const float* in_ptr=input.ptr<float>();
  const float* gamma_ptr=gamma.ptr<float>();
  const float* beta_ptr=beta.ptr<float>();
  float* out_ptr=const_cast<float*>(output.ptr<float>());
  const int32_t dim=static_cast<int32_t>(input.size());
  const float eps=1e-5f;
  float mean=0.0f;
  for(int32_t i=0;i<dim;++i) mean+=in_ptr[i];
  mean/=static_cast<float>(dim);
  float var=0.0f;
  for(int32_t i=0;i<dim;++i){ float d=in_ptr[i]-mean; var+=d*d; }
  var/=static_cast<float>(dim);
  const float inv_std=1.0f/std::sqrt(var+eps);
  for(int32_t i=0;i<dim;++i) out_ptr[i]=gamma_ptr[i]*(in_ptr[i]-mean)*inv_std+beta_ptr[i];
}
}  // namespace kernel
