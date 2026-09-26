#include "model/gpt2.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "base/tick.h"
namespace model {

void GPT2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) { add_layer_->set_cuda_config(config); add_layer_->to_cuda(); }
  if (gelu_layer_) { gelu_layer_->set_cuda_config(config); gelu_layer_->to_cuda(); }
  if (cls_layer_) { cls_layer_->set_cuda_config(config); cls_layer_->to_cuda(); }
  if (embedding_layer_) { embedding_layer_->set_cuda_config(config); embedding_layer_->to_cuda(); }
  if (pos_embedding_layer_) { pos_embedding_layer_->set_cuda_config(config); pos_embedding_layer_->to_cuda(); }
  if (mha_layer_) { mha_layer_->set_cuda_config(config); mha_layer_->to_cuda(); }
  auto migrate_vec = [&](std::vector<std::shared_ptr<op::Layer>>& vec) {
    for (auto& layer : vec) { if (layer) { layer->set_cuda_config(config); layer->to_cuda(); } }
  };
  migrate_vec(wq_layers_); migrate_vec(wk_layers_); migrate_vec(wv_layers_);
  migrate_vec(wo_layers_); migrate_vec(w1_layers_); migrate_vec(w2_layers_);
  migrate_vec(layernorm_layers_);
}

GPT2Model::GPT2Model(base::TokenizerType tokenizer_type, std::string token_path,
                     std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeGPT2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status GPT2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) return error::PathNotValid(token_path_);
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_)
    return error::InternalError("The cpu device do not support int8 quant model.");
  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return error::InternalError("The cuda handle create failed.");
  }
  Status read_status = gen_model_from_file();
  if (!read_status) return read_status;
  init_mem();
  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status GPT2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                int& next) const {
  if (input.is_empty()) return base::error::InvalidArgument("The input tensor is empty.");
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_)
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_layernorm(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

void GPT2Model::create_nonparam_layers() {
  CHECK(gpt2_layers_ != nullptr);
  gpt2_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);
  gpt2_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  gpt2_layers_->gelu_layer_ = std::make_shared<op::GELULayer>(device_type_, config_->hidden_dim_);
}

void GPT2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(gpt2_layers_ != nullptr);
  // 与 tools/export_gpt2.py 的 int8 布局一一对应。
  // RawModelDataInt8::weight 偏移单位是【字节】，fp32 部分要乘 sizeof(float)。
  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // wte（fp32）
  gpt2_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, dim, config_->seq_len_, std::abs(config_->vocab_size_));
  gpt2_layers_->embedding_layer_->set_weight(
      0, {std::abs(config_->vocab_size_), dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  pos += static_cast<size_t>(config_->vocab_size_) * dim * sizeof(float);

  // wpe（fp32）
  gpt2_layers_->pos_embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, dim, config_->seq_len_, config_->seq_len_);
  gpt2_layers_->pos_embedding_layer_->set_weight(
      0, {config_->seq_len_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  pos += static_cast<size_t>(config_->seq_len_) * dim * sizeof(float);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    int32_t bias_dim = 0;
    // LayerNorm（attention）gamma+beta fp32
    auto ln_attn = std::make_shared<op::LayerNormLayer>(device_type_, dim);
    ln_attn->set_weight(0, {dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * sizeof(float);
    ln_attn->set_weight(1, {dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * sizeof(float);
    gpt2_layers_->layernorm_layers_.push_back(ln_attn);

    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * dim + wq->get_scale_num() * sizeof(float);
    bias_dim = dim;
    wq->set_bias(0, bias_dim, (void*)this->raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim * sizeof(float);
    gpt2_layers_->wq_layers_.push_back(wq);

    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(config_->kv_dim_) * dim + wk->get_scale_num() * sizeof(float);
    bias_dim = config_->kv_dim_;
    wk->set_bias(0, bias_dim, (void*)this->raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim * sizeof(float);
    gpt2_layers_->wk_layers_.push_back(wk);

    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(config_->kv_dim_) * dim + wv->get_scale_num() * sizeof(float);
    bias_dim = config_->kv_dim_;
    wv->set_bias(0, bias_dim, (void*)this->raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim * sizeof(float);
    gpt2_layers_->wv_layers_.push_back(wv);

    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * dim + wo->get_scale_num() * sizeof(float);
    bias_dim = dim;
    wo->set_bias(0, bias_dim, (void*)this->raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim * sizeof(float);
    gpt2_layers_->wo_layers_.push_back(wo);

    // LayerNorm（FFN）gamma+beta fp32
    auto ln_ffn = std::make_shared<op::LayerNormLayer>(device_type_, dim);
    ln_ffn->set_weight(0, {dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * sizeof(float);
    ln_ffn->set_weight(1, {dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * sizeof(float);
    gpt2_layers_->layernorm_layers_.push_back(ln_ffn);

    int32_t hidden_dim = config_->hidden_dim_;
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * hidden_dim + w1->get_scale_num() * sizeof(float);
    bias_dim = hidden_dim;
    w1->set_bias(0, bias_dim, (void*)this->raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim * sizeof(float);
    gpt2_layers_->w1_layers_.push_back(w1);

    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * hidden_dim + w2->get_scale_num() * sizeof(float);
    bias_dim = dim;
    w2->set_bias(0, bias_dim, (void*)this->raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim * sizeof(float);
    gpt2_layers_->w2_layers_.push_back(w2);
  }

  auto ln_final = std::make_shared<op::LayerNormLayer>(device_type_, dim);
  ln_final->set_weight(0, {dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  pos += static_cast<size_t>(dim) * sizeof(float);
  ln_final->set_weight(1, {dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  pos += static_cast<size_t>(dim) * sizeof(float);
  gpt2_layers_->layernorm_layers_.push_back(ln_final);

  // cls：tied 复用 fp32 wte，必须用非量化层（否则把 fp32 当 int8 解析）
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
  if (config_->is_shared_weight_) {
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(0),
                          cpu_device_type);
  } else {
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  }
  gpt2_layers_->cls_layer_ = cls_layer;
}

void GPT2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(gpt2_layers_ != nullptr);
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  int32_t dim = config_->dim_;
  int32_t hidden_dim = config_->hidden_dim_;

  gpt2_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, dim, config_->seq_len_, std::abs(config_->vocab_size_));
  gpt2_layers_->embedding_layer_->set_weight(
      0, {std::abs(config_->vocab_size_), dim}, raw_model_data_->weight(0), cpu_device_type);

  size_t pos = static_cast<size_t>(config_->vocab_size_) * dim;
  gpt2_layers_->pos_embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, dim, config_->seq_len_, config_->seq_len_);
  gpt2_layers_->pos_embedding_layer_->set_weight(
      0, {config_->seq_len_, dim}, raw_model_data_->weight(pos), cpu_device_type);
  pos += static_cast<size_t>(config_->seq_len_) * dim;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    int32_t bias_dim = 0;
    auto ln_attn = std::make_shared<op::LayerNormLayer>(device_type_, dim);
    ln_attn->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    ln_attn->set_weight(1, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    gpt2_layers_->layernorm_layers_.push_back(ln_attn);

    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
    wq->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * dim;
    bias_dim = dim;
    wq->set_bias(0, bias_dim, (void*)raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim;
    gpt2_layers_->wq_layers_.push_back(wq);

    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
    wk->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(config_->kv_dim_) * dim;
    bias_dim = config_->kv_dim_;
    wk->set_bias(0, bias_dim, (void*)raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim;
    gpt2_layers_->wk_layers_.push_back(wk);

    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
    wv->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(config_->kv_dim_) * dim;
    bias_dim = config_->kv_dim_;
    wv->set_bias(0, bias_dim, (void*)raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim;
    gpt2_layers_->wv_layers_.push_back(wv);

    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
    wo->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * dim;
    bias_dim = dim;
    wo->set_bias(0, bias_dim, (void*)raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim;
    gpt2_layers_->wo_layers_.push_back(wo);

    auto ln_ffn = std::make_shared<op::LayerNormLayer>(device_type_, dim);
    ln_ffn->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    ln_ffn->set_weight(1, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    gpt2_layers_->layernorm_layers_.push_back(ln_ffn);

    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, false, true);
    w1->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(hidden_dim) * dim;
    bias_dim = hidden_dim;
    w1->set_bias(0, bias_dim, (void*)raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim;
    gpt2_layers_->w1_layers_.push_back(w1);

    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, false, true);
    w2->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * hidden_dim;
    bias_dim = dim;
    w2->set_bias(0, bias_dim, (void*)raw_model_data_->weight(pos), cpu_device_type);
    pos += bias_dim;
    gpt2_layers_->w2_layers_.push_back(w2);
  }

  auto ln_final = std::make_shared<op::LayerNormLayer>(device_type_, dim);
  ln_final->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
  pos += dim;
  ln_final->set_weight(1, {dim}, raw_model_data_->weight(pos), cpu_device_type);
  pos += dim;
  gpt2_layers_->layernorm_layers_.push_back(ln_final);

  gpt2_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    gpt2_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          raw_model_data_->weight(0), cpu_device_type);
  } else {
    gpt2_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          raw_model_data_->weight(pos), cpu_device_type);
  }
}

void GPT2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) alloc = base::CPUDeviceAllocatorFactory::get_instance();
  else alloc = base::CUDADeviceAllocatorFactory::get_instance();
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    gpt2_layers_->to_cuda(cuda_config_);
  }
  std::shared_ptr<base::DeviceAllocator> alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  tensor::Tensor ln_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, ln_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, ln_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, ln_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, ln_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));

  tensor::Tensor gelu_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kGeluOutput, gelu_output));

  tensor::Tensor pos_embedding(base::DataType::kDataTypeFp32, config_->seq_len_, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kPositionEmbedding, pos_embedding));
  tensor::Tensor pos_tokens(base::DataType::kDataTypeInt32, config_->seq_len_, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kPositionTokens, pos_tokens));

  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status GPT2Model::create_layers() {
  using namespace base;
  if (!gpt2_layers_) gpt2_layers_ = std::make_unique<GPT2Layers>();
  if (!is_quant_model_) create_param_layers(); else create_param_quant_layers();
  create_nonparam_layers();
  if (!gpt2_layers_->embedding_layer_) return error::InternalError("Create the embedding layer failed!");
  if (!gpt2_layers_->pos_embedding_layer_) return error::InternalError("Create the position embedding layer failed!");
  if (gpt2_layers_->layernorm_layers_.size() != 2 * config_->layer_num_ + 1) return error::InternalError("Create the layernorm layers failed!");
  if (gpt2_layers_->wq_layers_.size() != config_->layer_num_ || gpt2_layers_->wk_layers_.size() != config_->layer_num_ ||
      gpt2_layers_->wv_layers_.size() != config_->layer_num_ || gpt2_layers_->wo_layers_.size() != config_->layer_num_)
    return error::InternalError("Create the attention matmul layers failed.");
  if (gpt2_layers_->w1_layers_.size() != config_->layer_num_ || gpt2_layers_->w2_layers_.size() != config_->layer_num_)
    return error::InternalError("Create the FFN matmul layers failed.");
  if (!gpt2_layers_->add_layer_) return error::InternalError("Create the add layer failed!");
  if (!gpt2_layers_->mha_layer_) return error::InternalError("Create the mha layer failed!");
  if (!gpt2_layers_->gelu_layer_) return error::InternalError("Create the GELU layer failed!");
  return error::Success();
}

op::EmbeddingOutput GPT2Model::embedding(const std::vector<int>& tokens) const {
  return embedding_at(tokens, 0);
}

op::EmbeddingOutput GPT2Model::embedding_at(const std::vector<int>& tokens, int32_t start_pos) const {
  CHECK(gpt2_layers_ != nullptr);
  LOG_IF(FATAL, !gpt2_layers_->embedding_layer_) << "The embedding layer in the GPT-2 model is null pointer.";
  const int32_t token_num = static_cast<int32_t>(tokens.size());
  const int32_t dim = config_->dim_;
  LOG_IF(FATAL, token_num <= 0 || start_pos + token_num > config_->seq_len_)
      << "The token num/start pos is out of range: " << token_num << ", " << start_pos;

  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  input_tokens.reshape({token_num});
  input_embeddings.reshape({token_num, dim});
  for (int32_t i = 0; i < token_num; ++i) input_tokens.index<int32_t>(i) = tokens.at(i);
  auto token_num_tensor = tensor::Tensor(base::DataType::kDataTypeInt32, token_num);
  STATUS_CHECK(gpt2_layers_->embedding_layer_->forward(input_tokens, token_num_tensor, input_embeddings));

  auto pos_tokens = get_buffer(ModelBufferType::kPositionTokens);
  auto pos_embeddings = get_buffer(ModelBufferType::kPositionEmbedding);
  pos_tokens.reshape({token_num});
  pos_embeddings.reshape({token_num, dim});
  for (int32_t i = 0; i < token_num; ++i) pos_tokens.index<int32_t>(i) = start_pos + i;
  auto pos_num_tensor = tensor::Tensor(base::DataType::kDataTypeInt32, token_num);
  STATUS_CHECK(gpt2_layers_->pos_embedding_layer_->forward(pos_tokens, pos_num_tensor, pos_embeddings));

  CHECK_NE(gpt2_layers_->add_layer_, nullptr);
  input_embeddings.reshape({token_num * dim});
  pos_embeddings.reshape({token_num * dim});
  STATUS_CHECK(gpt2_layers_->add_layer_->forward(input_embeddings, pos_embeddings, input_embeddings));
  input_embeddings.reshape({token_num, dim});

  op::EmbeddingOutput output(input_tokens, input_embeddings, token_num_tensor);
  return output;
}

void GPT2Model::attention_layernorm(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(gpt2_layers_ != nullptr);
  tensor::Tensor ln_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  // layernorm_layers_ 按 [ln_attn, ln_ffn] 交错，第 layer_idx 层的 attention LN 下标 = 2*layer_idx
  std::shared_ptr<op::Layer> ln_layer = gpt2_layers_->layernorm_layers_.at(2 * layer_idx);
  if (!ln_layer) LOG(FATAL) << "The attention layernorm layer is a null pointer in the GPT-2 model";
  STATUS_CHECK(ln_layer->forward(input, ln_output));
}

void GPT2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(gpt2_layers_ != nullptr);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  auto ln_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  const auto& query_layer = gpt2_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr);
  STATUS_CHECK(query_layer->forward(ln_output, query));
  const auto& key_layer = gpt2_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr);
  STATUS_CHECK(key_layer->forward(ln_output, key));
  const auto& value_layer = gpt2_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr);
  STATUS_CHECK(value_layer->forward(ln_output, val));
  // GPT-2 不使用 RoPE
}

void GPT2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(gpt2_layers_ != nullptr);
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  const auto& mha_layer = gpt2_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr);
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = gpt2_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void GPT2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(gpt2_layers_ != nullptr);
  CHECK_NE(gpt2_layers_->add_layer_, nullptr);
  STATUS_CHECK(gpt2_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  // 第 layer_idx 层的 FFN LayerNorm 下标 = 2*layer_idx + 1
  const auto& ffn_ln = gpt2_layers_->layernorm_layers_.at(2 * layer_idx + 1);
  CHECK_NE(ffn_ln, nullptr);
  STATUS_CHECK(ffn_ln->forward(input, ffn_norm_output));
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = gpt2_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr);
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
  tensor::Tensor gelu_output = get_buffer(ModelBufferType::kGeluOutput);
  CHECK_NE(gpt2_layers_->gelu_layer_, nullptr);
  STATUS_CHECK(gpt2_layers_->gelu_layer_->forward(w1_output, gelu_output));
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = gpt2_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr);
  STATUS_CHECK(w2_layer->forward(gelu_output, w2_output));
  CHECK_NE(gpt2_layers_->add_layer_, nullptr);
  STATUS_CHECK(gpt2_layers_->add_layer_->forward(input, w2_output, input));
}

void GPT2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(gpt2_layers_ != nullptr);
  const auto& ln_final = gpt2_layers_->layernorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(ln_final, nullptr);
  STATUS_CHECK(ln_final->forward(input, input));
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(gpt2_layers_->cls_layer_, nullptr);
  STATUS_CHECK(gpt2_layers_->cls_layer_->forward(input, forward_output));
}

base::Status GPT2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) return status;
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

int32_t GPT2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();
  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

}  // namespace model
