
#ifndef KUIPER_INCLUDE_OP_SIMPLE_GPT2_ENCODE_H_
#define KUIPER_INCLUDE_OP_SIMPLE_GPT2_ENCODE_H_
#include "op/encode.h"
#include <cstdint>
#include <string>
#include <vector>
#include "base/gpt2_bpe.h"
namespace op {
class SimpleGPT2EncodeLayer : public EncodeLayerBase {
 public:
  explicit SimpleGPT2EncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
  std::vector<int32_t> encode(const std::string& sentence) const override;
  std::string decode(int32_t token_id) const override;
  std::string decode(const std::vector<int32_t>& token_ids) const override;
  bool is_sentence_ending(int32_t token_id) const override;
  int32_t vocab_size() const override;
 private:
  base::gpt2bpe::GPT2Bpe bpe_;
  int32_t eos_id_ = 50256;
  int32_t num_token_ = 50257;
};
}  // namespace op
#endif
