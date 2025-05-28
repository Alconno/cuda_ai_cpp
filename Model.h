#pragma once

#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include <variant>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <thread>

class ModelConfig {
public:
    int vocab_size = 30;
    int block_size = 12;
    int n_embd = 32;
    int n_layer = 1;
    int n_head = 2;
    bool bias = false;
    dt dropout = 0.0f;

    dt emb_std = 0.02f;
    dt attn_std = 0.1f;
    dt mlp_std = std::sqrt(2.0f / n_embd);
    dt ln_gamma = 1.0f;
    dt linear_std = 0.5f; // lm_head std dev

    ModelConfig();
    ModelConfig(int vocab_size, int block_size, int n_embd, int n_layer, int n_head,
        bool bias = false, dt dropout = 0.0f, dt emb_std = 0.02f,
        dt attn_std = 0.1f, dt mlp_std = 0.1f, dt ln_gamma = 1.0f, dt linear_std = 0.5f);
};


class LayerNorm : public Layer {
public:
    ModelConfig config;
    std::unordered_map<std::string, Tensor> tmem;

    int features;
    dt eps;

    LayerNorm(ModelConfig config, dt eps = 1e-6);
    LayerNorm();

    void initialize_weights();
    void initialize_mem();
    void forward(Tensor& x);
    void update(const dt& lr);
    void set_gamma(std::vector<dt>& data);
    void set_beta(std::vector<dt>& data);
    void to_gpu();
    void to_cpu();
};


class SelfAttention : public Layer {
public:
    ModelConfig config;
    std::unordered_map<std::string, Tensor> tmem;

    Linear c_attn = Linear();
    Linear c_proj = Linear();

    SelfAttention(ModelConfig config);
    SelfAttention();

    void initialize_mem(int B, int T, int C, int n_head);
    void forward(Tensor& x);
    void update(const dt& lr);
    void to_gpu();
    void to_cpu();
};


class MLP : public Layer {
public:
    ModelConfig config;
    Linear mlp = Linear();
    Linear mlp_proj = Linear();
    std::unordered_map<std::string, Tensor> tmem;

    MLP(ModelConfig config);
    MLP();

    void initialize_mem(int B, int T, int C);
    void forward(Tensor& x);
    void update(const dt& lr);
    void to_gpu();
    void to_cpu();
};


class Block : public Layer {
public:
    std::unordered_map<std::string, std::shared_ptr<Layer>> layers;
    LayerNorm* ln_0;
    LayerNorm* ln_1;
    SelfAttention* attn;
    MLP* mlp;

    Block(ModelConfig config);

    void forward(Tensor*& x);
    void update(const dt& lr);
    void to_gpu();
    void to_cpu();
};


class LayerList : public Layer {
public:
    std::vector<std::shared_ptr<Block>> blocks;

    LayerList(ModelConfig config);

    void to_gpu();
    void to_cpu();
};


class Model {
public:
    ModelConfig config;
    std::unordered_map<std::string, std::shared_ptr<Layer>> transformer;
    std::shared_ptr<Linear> lm_head = std::make_shared<Linear>(Linear());

    LayerNorm ln0;
    SelfAttention attn;

    bool cuda_enabled = false;

    Model(ModelConfig config, bool cuda = false);

    void train(const int& max_steps, const Optimizer& opt, const int& B);

    void print_predictions(Tensor& logits_softmax_out, const std::vector<int>& logits_shape, const std::vector<int>& targets);

    void save_model(std::string model_name);
    void load_model(std::string model_name);

    void to_gpu();
    void to_cpu();

    int64_t count_parameters() const {
        int64_t total_params = 0;

        // Embeddings
        total_params += static_cast<int64_t>(config.vocab_size) * config.n_embd;   // token embedding
        total_params += static_cast<int64_t>(config.block_size) * config.n_embd;   // positional embedding

        for (int i = 0; i < config.n_layer; i++) {
            // Attention: Q, K, V projections and output projection
            // Each linear layer: weights (n_embd*n_embd), optionally bias (n_embd)
            int64_t linear_weights = static_cast<int64_t>(config.n_embd) * config.n_embd;
            int64_t linear_bias = config.bias ? config.n_embd : 0;

            // 3 projections for Q,K,V
            total_params += 3 * (linear_weights + linear_bias);
            // Output projection
            total_params += linear_weights + linear_bias;

            // LayerNorm (2 per layer: pre-attn and pre-MLP)
            // Each LayerNorm has gamma and beta vectors of length n_embd
            total_params += 2 * 2 * config.n_embd;

            // MLP: usually two linear layers: first expands to 4*n_embd, second projects back
            int64_t mlp_in = config.n_embd;
            int64_t mlp_hidden = 4 * config.n_embd;

            // First linear layer weights + bias
            total_params += mlp_in * mlp_hidden;
            total_params += config.bias ? mlp_hidden : 0;

            // Second linear layer weights + bias
            total_params += mlp_hidden * mlp_in;
            total_params += config.bias ? mlp_in : 0;
        }

        // Final LM head
        total_params += static_cast<int64_t>(config.n_embd) * config.vocab_size;
        total_params += config.bias ? config.vocab_size : 0;

        return total_params;
    }
};

#endif // MODEL_H
