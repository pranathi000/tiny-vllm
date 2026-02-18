#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "kernels.cuh"

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

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed\n";
        return 1;
    }
    std::cout << "cuBLAS initialized OK\n";
    cublasDestroy(handle);

    return 0;
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
#ifdef DEBUG
    std::cout << "Copied model weights from CPU to GPU correctly!\n";
    std::cout << "Checking if model weights are copied from CPU to GPU correctly:";
    std::vector<char> test_from_gpu;
    test_from_gpu.resize(20);
    cudaMemcpy(test_from_gpu.data(), model_weights, 20, cudaMemcpyDeviceToHost);
    std::cout << "\nCopied from GPU:\n";
    std::cout << "\n"
              << test_from_gpu.data() << "\n";
    for (auto &i : test_from_gpu)
    {
        printf("%02x ", (unsigned char)i);
    }
    std::cout << "\nOriginal CPU data:\n";
    for (int i = 0; i < 20; ++i)
    {
        printf("%02x ", (unsigned char)model_weights_cpu[i]);
    }
#endif
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
    std::cout << "\nInput tokens copied to GPU:\n";
    std::vector<int> test_from_gpu_tokens;
    test_from_gpu_tokens.resize(input_tokens.size());
    cudaMemcpy(test_from_gpu_tokens.data(), gpu_input_tokens, input_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "\nCopied tokens from GPU:\n";
    for (auto &i : test_from_gpu_tokens)
    {
        std::cout << i << "\n";
    }
    std::cout << "\nOriginal CPU tokens data:\n";
    for (int i = 0; i < input_tokens.size(); ++i)
    {
        std::cout << input_tokens[i] << "\n";
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
    std::cout << "Checking if the correct embedding was copied" << std::endl;
    std::vector<__nv_bfloat16> test_gpu_input_embeds;
    test_gpu_input_embeds.resize(EMBEDDING_LENGTH * input_tokens.size());
    cudaMemcpy(test_gpu_input_embeds.data(), input_embeddings, input_tokens.size() * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, cudaMemcpyDeviceToHost);
    std::cout << "\nCopied embeds from GPU:\n";
    std::cout << "\nOriginal CPU embeds data:\n";
    std::cout << "model_weights_cpu.size():" << model_weights_cpu.size() << std::endl;
    for (int token = 0; token < input_tokens.size(); ++token)
    {
        for (int i = 0; i < 2048; ++i)
        {
            __nv_bfloat16 *all_embeds_cpu = (__nv_bfloat16 *)(model_weights_cpu.data() + offsets.at("model.embed_tokens.weight"));
            // std::cout << "From gpu:" << (float)test_gpu_input_embeds[token * 2048 + i] << "- from cpu: " << (float)all_embeds_cpu[input_tokens[token] * 2048 + i] << "\n";
        }
    }
#endif
    std::cout << "\nOk bye!\n";
    return 0;
}