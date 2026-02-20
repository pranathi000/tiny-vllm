#pragma once

bool verifyEmbeddingGather(std::vector<int> &input_tokens, nv_bfloat16 *input_embeddings, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets);

bool verifyRMSNormWeights(std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets);

bool verifyQProjection(cublasStatus_t gemm_status, std::vector<int> &input_tokens, nv_bfloat16 *q, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets, nv_bfloat16 *rms_norms);

bool verifyInputTokensCopy(std::vector<int> &input_tokens, int *gpu_input_tokens);

bool verifyModelWeightsCopy(void *model_weights, std::vector<char> &model_weights_cpu);
