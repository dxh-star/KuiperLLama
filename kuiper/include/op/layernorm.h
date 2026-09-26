
#ifndef KUIPER_INCLUDE_OP_LAYERNORM_H_
#define KUIPER_INCLUDE_OP_LAYERNORM_H_
#include "layer.h"
namespace op {
class LayerNormLayer : public LayerParam {
 public:
  explicit LayerNormLayer(base::DeviceType device_type, int32_t dim);
  base::Status forward() override;
  base::Status check() const override;
 private:
  int32_t dim_ = 0;
};
}  // namespace op
#endif
