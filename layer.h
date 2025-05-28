#pragma once
#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <cassert>
#include <random>
#include "tensor.h"

using dt = float;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Layer {
public:
    Tensor weights;

    Layer() = default;
    virtual ~Layer() = default;
};

class Dropout : public Layer {
private:
    float dropout_rate;
    std::default_random_engine generator;
    std::bernoulli_distribution distribution;

public:
    explicit Dropout(float rate);
    void forward(Tensor& input);
    Tensor get_weights() const;
};

class Embedding : public Layer {
private:
    int input_dim;   // should be int, not dt
    int output_dim;  // should be int, not dt
    double std_dev;
    double mean_val;

public:
    Embedding(int input_dim, int output_dim, double std_dev = 0.2, double mean_val = 0.0);
    void initialize_weights();
    void forward(std::vector<int>& input_indices, Tensor& out, int B, int T);

    void update(dt lr);
    void to_gpu();
    void to_cpu();
    Tensor get_weights() const;
};

class Linear : public Layer {
public:
    Tensor bias_v;
    Tensor matmul_result;
    int input_dim;
    int output_dim;
    double std_dev;
    double mean_val;
    bool bias;

    Linear(int input_dim, int output_dim, bool bias = false, double std_dev = 0.25, double mean_val = 0.0);
    Linear();

    void initialize_weights();
    void forward(Tensor& input, Tensor& output);
    void update(dt lr);
    void to_gpu();
    void to_cpu();

    Tensor get_weights() const;
    void set_weights(const std::vector<dt>& data);
    Tensor get_bias() const;
    void set_bias(const std::vector<dt>& data);
    void zero_grad();
};

class Optimizer {
public:
    int max_steps;
    int warmup_steps;
    dt max_lr;
    dt min_lr;

    Optimizer(int max_steps_, int warmup_steps_, dt max_lr_, dt min_lr_)
        : max_steps(max_steps_), warmup_steps(warmup_steps_), max_lr(max_lr_), min_lr(min_lr_) {}
};

std::string getWordBetweenDelimiters(const std::string& input, int startIndex, int endIndex);

int extract_number_between(const std::string& text, const std::string& start, const std::string& end);

dt get_lr(const int& step, const Optimizer& opt);

#endif // LAYER_H
