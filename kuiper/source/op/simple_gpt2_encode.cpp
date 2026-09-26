
#include "op/simple_gpt2_encode.h"
#include <glog/logging.h>
#include <string>
namespace op {
SimpleGPT2EncodeLayer::SimpleGPT2EncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
  LOG_IF(FATAL, !bpe_.load(token_model_path_))
      << "Load the GPT-2 tokenizer failed, the path is: " << token_model_path_;
  num_token_ = bpe_.vocab_size();
  LOG(INFO) << "GPT-2 tokenizer loaded, vocab size: " << num_token_;
}
std::vector<int32_t> SimpleGPT2EncodeLayer::encode(const std::string& sentence) const {
  std::vector<int32_t> tokens = bpe_.encode(sentence);
  if (has_bos_) tokens.insert(tokens.begin(), eos_id_);
  if (has_eos_) tokens.push_back(eos_id_);
  return tokens;
}
std::string SimpleGPT2EncodeLayer::decode(int32_t token_id) const { return bpe_.decode_token(token_id); }
std::string SimpleGPT2EncodeLayer::decode(const std::vector<int32_t>& token_ids) const { return bpe_.decode(token_ids); }
bool SimpleGPT2EncodeLayer::is_sentence_ending(int32_t token_id) const { return token_id == eos_id_; }
int32_t SimpleGPT2EncodeLayer::vocab_size() const { return num_token_; }
}  // namespace op
