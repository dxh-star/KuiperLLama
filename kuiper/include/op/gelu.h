
#ifndef KUIPER_INCLUDE_OP_GELU_H_
#define KUIPER_INCLUDE_OP_GELU_H_
#include "layer.h"
namespace op {
class GELULayer : public Layer {
 public:
  explicit GELULayer(base::DeviceType device_type, int32_t dim);
  base::Status forward() override;
  base::Status check() const override;
 private:
  int32_t dim_ = 0;
};
}  // namespace op
#endif
