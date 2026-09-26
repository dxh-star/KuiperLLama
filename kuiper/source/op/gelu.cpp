
#include "op/gelu.h"
#include "kernels/cpu/gelu_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
GELULayer::GELULayer(base::DeviceType device_type, int32_t dim)
    : Layer(device_type, LayerType::kLayerGELU, "GELU"), dim_(dim) {
  reset_input_size(1); reset_output_size(1);
}
base::Status GELULayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
  if (!status) { LOG(ERROR) << "The input tensor error in the gelu layer."; return status; }
  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
  if (!status) { LOG(ERROR) << "The output tensor error in the gelu layer."; return status; }
  return base::error::Success();
}
base::Status GELULayer::forward() {
  auto status = check();
  if (!status) return status;
  auto input = this->get_input(0);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) CHECK(cuda_config_ != nullptr);
  kernel::get_gelu_kernel(device_type_)(input, output, cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}
}  // namespace op
