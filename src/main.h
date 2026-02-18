#pragma once

bool verifyEmbeddingGather(std::vector<int> &input_tokens, nv_bfloat16 *input_embeddings, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets);

bool verifyInputTokensCopy(std::vector<int> &input_tokens, int *gpu_input_tokens);

bool verifyModelWeightsCopy(void *model_weights, std::vector<char> &model_weights_cpu);
