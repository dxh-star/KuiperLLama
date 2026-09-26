#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/qwen2.h"
int32_t generate(const model::Qwen2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  words.push_back(next);
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}


int main(int argc, char* argv[]) {
  if (argc < 3) {
    LOG(INFO) << "Usage: ./qwen_infer checkpoint_path tokenizer_path [prompt] [quant]";
    LOG(INFO) << "  quant: checkpoint 是 int8 量化导出的模型 (export_qwen_hf.py --version 3)";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  std::string sentence = "你好，请介绍一下你自己。";
  bool is_quant_model = false;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "quant") {
      is_quant_model = true;
    } else if (arg == "cpu") {
      LOG(FATAL) << "Qwen int8 不支持 CPU；FP32 请在普通 build（无 QWEN2_SUPPORT）下运行";
    } else {
      sentence = arg;
    }
  }
  LOG(INFO) << "Load Qwen2 model, quant model: " << (is_quant_model ? "true" : "false");

  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
    checkpoint_path, is_quant_model);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }

  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps:%d\n", steps);
  printf("\nduration:%lf\n", duration);
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}