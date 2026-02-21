#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "kernels.cuh"
#include "main.h"

using json = nlohmann::json;

constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;

int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    return 0;
}

bool verifyModelWeightsCopy(void *model_weights, std::vector<char> &model_weights_cpu)
{
    std::vector<char> test_from_gpu;
    test_from_gpu.resize(20);
    cudaMemcpy(test_from_gpu.data(), model_weights, 20, cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for (int i = 0; i < 20; ++i)
    {
        if ((unsigned char)model_weights_cpu[i] == (unsigned char)test_from_gpu[i])
        {
            continue;
        }
        if (is_correct)
        {
            std::cout << "Model weights copied to GPU incorrectly!:\n";
        }
        printf("%02x ", (unsigned char)model_weights_cpu[i] == (unsigned char)test_from_gpu[i]);
        is_correct = false;
    }
    return is_correct;
}

bool verifyInputTokensCopy(std::vector<int> &input_tokens, int *gpu_input_tokens)
{
    std::vector<int> test_from_gpu_tokens;
    test_from_gpu_tokens.resize(input_tokens.size());
    cudaMemcpy(test_from_gpu_tokens.data(), gpu_input_tokens, input_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for (int i = 0; i < input_tokens.size(); ++i)
    {
        if (input_tokens[i] == test_from_gpu_tokens[i])
        {
            continue;
        }
        if (is_correct)
        {
            std::cout << "Input tokens copy mismatch!" << std::endl;
        }
        std::cout << "CPU: " << input_tokens[i] << " | GPU: " << test_from_gpu_tokens[i] << "\n";
        is_correct = false;
    }
    return is_correct;
}

bool verifyEmbeddingGather(std::vector<int> &input_tokens, nv_bfloat16 *input_embeddings, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets)
{
    std::vector<__nv_bfloat16> test_gpu_input_embeds;
    test_gpu_input_embeds.resize(EMBEDDING_LENGTH * input_tokens.size());
    cudaMemcpy(test_gpu_input_embeds.data(), input_embeddings, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for (int token = 0; token < input_tokens.size(); ++token)
    {
        for (int i = 0; i < 2048; ++i)
        {
            __nv_bfloat16 *all_embeds_cpu = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.embed_tokens.weight"));
            if ((float)test_gpu_input_embeds[token * 2048 + i] != (float)all_embeds_cpu[input_tokens[token] * 2048 + i])
            {
                if (is_correct)
                {
                    std::cout << "Incorrect embeddings were retrieved" << std::endl;
                }
                std::cout << "GPU:" << (float)test_gpu_input_embeds[token * 2048 + i] << " | CPU: " << (float)all_embeds_cpu[input_tokens[token] * 2048 + i] << "\n";
                is_correct = false;
            }
        }
    }
    return is_correct;
}

bool floats_close_enough(float a, float b)
{
    return fabs(a - b) / fmax(fabs(a), fabs(b)) < 1e-3;
}

bool verifyRMSNormWeights(std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets)
{
    __nv_bfloat16 *layernorm_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.layers.0.input_layernorm.weight"));
    std::vector<float> rms_norm_debug_values = {0.154297, 0.182617, 0.255859, -0.0116577, 0.140625, 0.19043, -0.139648, -0.160156, 0.139648, -0.170898};
    bool is_correct_rms_weight = true;
    for (int i = 0; i < 10; ++i)
    {
        if (!floats_close_enough((float)layernorm_weights[i], rms_norm_debug_values[i]))
        {
            if (is_correct_rms_weight)
            {
                std::cout << "RMS norm weights check failed" << std::endl;
            }
            std::cout << "Expected RMS norm weight: " << rms_norm_debug_values[i] << ", received: " << (float)layernorm_weights[i] << std::endl;
            is_correct_rms_weight = false;
        }
    }
    return is_correct_rms_weight;
}

bool verifyRmsNorm(__nv_bfloat16 *gpu_input, __nv_bfloat16 *gpu_output,
                   std::vector<char> &model_weights_cpu,
                   std::unordered_map<std::string, uint64_t> &offsets,
                   int num_tokens, int layer)
{
    constexpr float EPSILON = 1e-5f;
    constexpr float TOLERANCE = 1e-2f;

    std::vector<__nv_bfloat16> cpu_input(num_tokens * EMBEDDING_LENGTH);
    std::vector<__nv_bfloat16> cpu_output(num_tokens * EMBEDDING_LENGTH);
    cudaMemcpy(cpu_input.data(), gpu_input, num_tokens * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_output.data(), gpu_output, num_tokens * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    std::string weight_key = "model.layers." + std::to_string(layer) + ".input_layernorm.weight";
    __nv_bfloat16 *norm_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at(weight_key));

    int mismatches = 0;
    for (int t = 0; t < num_tokens; ++t)
    {
        float sum_sq = 0.0f;
        for (int i = 0; i < EMBEDDING_LENGTH; ++i)
        {
            float val = (float)cpu_input[t * EMBEDDING_LENGTH + i];
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / EMBEDDING_LENGTH + EPSILON);

        for (int i = 0; i < EMBEDDING_LENGTH; ++i)
        {
            float input_val = (float)cpu_input[t * EMBEDDING_LENGTH + i];
            float weight_val = (float)norm_weights[i];
            float expected = (input_val / rms) * weight_val;
            float actual = (float)cpu_output[t * EMBEDDING_LENGTH + i];

            float rel_err = (expected == 0.0f) ? fabs(actual) : fabs(actual - expected) / fabs(expected);
            if (rel_err > TOLERANCE || isnanf(actual) || isnanf(expected))
            {
                if (mismatches < 10)
                {
                    std::cout << "RMSNorm MISMATCH token=" << t << " elem=" << i
                              << " expected=" << expected << " got=" << actual
                              << " rel_err=" << rel_err << "\n";
                }
                mismatches++;
            }
        }
    }

    return mismatches == 0;
}

bool verifyQProjection(cublasStatus_t gemm_status, std::vector<int> &input_tokens, nv_bfloat16 *q, std::vector<char> &model_weights_cpu, std::unordered_map<std::string, uint64_t> &offsets, nv_bfloat16 *rms_norms)
{
    std::cout << "Cublas first gemm status: " << gemm_status << std::endl;
    std::vector<__nv_bfloat16> q_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(q_cpu.data(), q, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> q_cpu_crosscheck(input_tokens.size() * EMBEDDING_LENGTH);
    __nv_bfloat16 *q_cpu_weights = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.layers.0.self_attn.q_proj.weight"));
    std::vector<__nv_bfloat16> rms_norms_cpu(input_tokens.size() * EMBEDDING_LENGTH);
    cudaMemcpy(rms_norms_cpu.data(), rms_norms, input_tokens.size() * EMBEDDING_LENGTH * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    // input_tokens * w_q^T (N, 2048) x (2048, 2048) -> (N, 2048)
    bool is_correct = true;
    for (int token_idx = 0; token_idx < input_tokens.size(); ++token_idx)
    {
        for (int j = 0; j < EMBEDDING_LENGTH; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < EMBEDDING_LENGTH; ++k)
            {
                float input_value = (float)rms_norms_cpu[token_idx * EMBEDDING_LENGTH + k];
                float weight_value = (float)q_cpu_weights[j * EMBEDDING_LENGTH + k];
                sum += input_value * weight_value;
            }
            float actual = (float)q_cpu[token_idx * EMBEDDING_LENGTH + j];
            float rel_err = (sum == 0.0f) ? fabs(actual) : fabs(actual - sum) / fabs(sum);
            if (rel_err > 1e-1)
            {
                std::cout << "Q MISMATCH token=" << token_idx << " dim=" << j
                          << " expected=" << sum << " got=" << actual
                          << " rel_err=" << rel_err << "\n";
                is_correct = false;
            }
        }
    }
    if (is_correct)
    {
        std::cout << "Q projection check done, all correct!" << std::endl;
    }
    else
    {
        std::cout << "Q projection check failed!" << std::endl;
    }
    return is_correct;
}

struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};

int main(int argc, char *argv[])
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    // READ SAFETENSORS
    std::ifstream safetensors_file("model.safetensors", std::ios_base::binary); // TODO: use args to provide the path or smth
    if (!safetensors_file.is_open())
    {
        std::cout << "Can't open model.safetensors file\n";
        safetensors_file.close();
        return 1;
    }

    // READ SAFETENSORS HEADER SIZE
    uint64_t header_size;
    // reinterpret_cast<char*>(&header_size) gives me an address of header_size
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
#ifdef DEBUG
    std::cout << "Safetensors header size read correctly. Size of header: " << header_size << std::endl;
#endif
    // READ SAFETENSORS HEADER
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);
#ifdef DEBUG
    std::cout << "Header read correctly\n";
#endif
    // READ OFFSETS OF EVERY LAYER (TENSOR) TO KNOW WHERE EVERY LAYER STARTS AND ENDS IN THE MEMORY
    std::unordered_map<std::string, uint64_t> offsets;
    json header_json = json::parse(header);
    uint64_t max_offset = 0;
    for (auto &[key, value] : header_json.items())
    {
        if (key == "__metadata__")
        {
            continue;
        }
        uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
        if (offset_end > max_offset)
        {
            max_offset = offset_end;
        }
        offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
    }

    void *model_weights;
    cudaMalloc(&model_weights, max_offset); // max_offset tells where the model weights end in the memory

    std::vector<char> model_weights_cpu;
    model_weights_cpu.resize(max_offset);
    safetensors_file.read(model_weights_cpu.data(), max_offset);

    cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
    if (!verifyModelWeightsCopy(model_weights, model_weights_cpu))
    {
        return 1;
    }
    safetensors_file.close();

    // BASICALLY A HELPER STRUCT TO HAVE AN EASY ACCESS TO ANY MODEL WEIGHTS ON GPU
    // TODO: right now I know the model structure since it's always llama 3.2 1B-Instruct, but maybe it would be convenient
    //       to store dimensions somewhere for even easier access?
    Weights weights{};
    weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.embed_tokens.weight"));
    weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.norm.weight"));
    for (int i = 0; i < N_LAYERS; ++i)
    {
        weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
        weights.mlp_down_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight"));
        weights.mlp_gate_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"));
        weights.mlp_up_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight"));
        weights.post_attn_layernorms[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"));
        weights.w_k[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight"));
        weights.w_o[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight"));
        weights.w_q[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight"));
        weights.w_v[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight"));
    }

    // LLM INPUT
    std::vector<int> input_tokens;
#ifdef DEBUG
    input_tokens.push_back(128000);
    input_tokens.push_back(791);
    input_tokens.push_back(6864);
    input_tokens.push_back(315);
    input_tokens.push_back(9822);
    input_tokens.push_back(374);
#else
    int token;
    while (std::cin >> token)
    {
        input_tokens.push_back(token);
    }
#endif
#ifdef DEBUG
    std::cout << "Input tokens:\n";
    for (auto &token : input_tokens)
    {
        std::cout << token << "\n";
    }
#endif

    int *gpu_input_tokens;
    cudaMalloc(&gpu_input_tokens, input_tokens.size() * sizeof(int));
    cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
#ifdef DEBUG
    if (!verifyInputTokensCopy(input_tokens, gpu_input_tokens))
    {
        return 1;
    }
#endif
    // INFERENCE STARTS HERE! =]
    // I have the same amount of embeddings as input tokens
    // it's just every embedding is 2048 length bf16 vector
    // retrieved from model weights based on token's value
    __nv_bfloat16 *input_embeddings;
    cudaMalloc(&input_embeddings, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, input_tokens.size());
    cudaDeviceSynchronize();
#ifdef DEBUG
    if (!verifyEmbeddingGather(input_tokens, input_embeddings, model_weights_cpu, offsets))
    {
        return 1;
    }
#endif
    __nv_bfloat16 *rms_norms;
    cudaMalloc(&rms_norms, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    rmsNorm(input_embeddings, rms_norms, weights.input_layernorm[0], input_tokens.size());
    cudaDeviceSynchronize();
#ifdef DEBUG
    if (!verifyRMSNormWeights(model_weights_cpu, offsets) || !verifyRmsNorm(input_embeddings, rms_norms, model_weights_cpu, offsets, input_tokens.size(), 0))
    {
        std::cout << "RMS norm verification failed" << std::endl;
        return 1;
    }
#endif

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }

#ifdef DEBUG
    std::cout << "cuBLAS initialized OK\n";
#endif

    // Q = inputs * wq^T; my matrices are row-major, cublas expects column-major
    // it perceives my matrices as transposed
    // there's a trick where C = A * B == C^T = B^T * A^T
    // so in my scenario cublas sees now: Q = inputs^T * wq^T^T = inputs ^T * wq
    // so I need to do: Q^T = wq ^T * inputs
    // the beauty is that we don't need to transpose Q^T back to Q
    // because cublas sees the output as column-major
    // so it's in fact transposed
    // final dim (num_tok, 2048)
    __nv_bfloat16 *q_proj;
    cudaMalloc(&q_proj, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    float q_proj_alpha = 1.0f;
    float q_proj_beta = 0.0f;
    auto q_proj_status = cublasGemmEx(cublas_handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      EMBEDDING_LENGTH,
                                      input_tokens.size(),
                                      EMBEDDING_LENGTH,
                                      &q_proj_alpha,
                                      weights.w_q[0],
                                      CUDA_R_16BF,
                                      EMBEDDING_LENGTH,
                                      rms_norms,
                                      CUDA_R_16BF,
                                      EMBEDDING_LENGTH,
                                      &q_proj_beta,
                                      q_proj,
                                      CUDA_R_16BF,
                                      EMBEDDING_LENGTH,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
#ifdef DEBUG
    verifyQProjection(q_proj_status, input_tokens, q_proj, model_weights_cpu, offsets, rms_norms);
#endif

    __nv_bfloat16 *k_proj;
    cudaMalloc(&k_proj, input_tokens.size() * sizeof(__nv_bfloat16) * 512);
    // input = (num_tokens, 2048), weights = (512, 2048)
    // after trick: (512, 2048) * (2048, num_tokens) -> (512, num_tokens), which really is (num_tok, 512)
    // lda: 2048, ldb: 2048, ldc: 512

    float k_proj_alpha = 1.0f;
    float k_proj_beta = 0.0f;
    auto k_proj_status = cublasGemmEx(cublas_handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      512,
                                      input_tokens.size(),
                                      EMBEDDING_LENGTH,
                                      &k_proj_alpha,
                                      weights.w_k[0],
                                      CUDA_R_16BF,
                                      EMBEDDING_LENGTH,
                                      rms_norms,
                                      CUDA_R_16BF,
                                      EMBEDDING_LENGTH,
                                      &k_proj_beta,
                                      k_proj,
                                      CUDA_R_16BF,
                                      512,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT);

    // same as K projection
    __nv_bfloat16 *v_proj;
    cudaMalloc(&v_proj, input_tokens.size() * sizeof(__nv_bfloat16) * 512);

    float v_proj_alpha = 1.0f;
    float v_proj_beta = 0.0f;
    auto v_proj_status = cublasGemmEx(cublas_handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      512,
                                      input_tokens.size(),
                                      EMBEDDING_LENGTH,
                                      &v_proj_alpha,
                                      weights.w_v[0],
                                      CUDA_R_16BF,
                                      EMBEDDING_LENGTH,
                                      rms_norms,
                                      CUDA_R_16BF,
                                      EMBEDDING_LENGTH,
                                      &v_proj_beta,
                                      v_proj,
                                      CUDA_R_16BF,
                                      512,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT);

    // RoPE now
    rope(q_proj, input_tokens.size(), 2048);
    rope(k_proj, input_tokens.size(), 512);
    cudaDeviceSynchronize();

    std::cout << "\nOk bye!\n";
    cublasDestroy(cublas_handle);
    cudaDeviceSynchronize();
    return 0;
}
