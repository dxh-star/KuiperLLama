
#include "op/layernorm.h"
#include <cuda_runtime_api.h>
#include <armadillo>
#include "kernels/cpu/layernorm_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
LayerNormLayer::LayerNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::kLayerLayerNorm, false, "LayerNorm"), dim_(dim) {
  reset_input_size(1); reset_output_size(1); reset_weight_size(2);
}
base::Status LayerNormLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto input = this->get_input(0);
  auto gamma = this->get_weight(0);
  auto beta = this->get_weight(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) CHECK(cuda_config_ != nullptr);
  kernel::get_layernorm_kernel(device_type_)(input, gamma, beta, output,
                                            cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}
base::Status LayerNormLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
  if (!status) { LOG(ERROR) << "The input tensor error in the layernorm layer."; return status; }
  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_);
  if (!status) { LOG(ERROR) << "The gamma tensor error in the layernorm layer."; return status; }
  status = check_tensor_with_dim(get_weight(1), device_type_, data_type_, dim_);
  if (!status) { LOG(ERROR) << "The beta tensor error in the layernorm layer."; return status; }
  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
  if (!status) { LOG(ERROR) << "The output tensor error in the layernorm layer."; return status; }
  return base::error::Success();
}
}  // namespace op
