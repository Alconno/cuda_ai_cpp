#include "layer.h"
#include "kernels.h"

// -------- Dropout --------

Dropout::Dropout(float rate)
    : dropout_rate(rate), distribution(1.0 - rate), generator(std::random_device{}()) {}

void Dropout::forward(Tensor& input) {
    float scale = 1.0f / (1.0f - dropout_rate);
    for (dt& data : input.data) {
        if (!distribution(generator)) {
            data = dt(0.0);
        }
        else {
            data *= scale;
        }
    }
}

Tensor Dropout::get_weights() const {
    return Tensor({});  // Dropout has no weights
}




// -------- Embedding --------

Embedding::Embedding(int input_dim_, int output_dim_, double std_dev_, double mean_val_)
    : input_dim(input_dim_), output_dim(output_dim_), std_dev(std_dev_), mean_val(mean_val_) {
    initialize_weights();
}

void Embedding::initialize_weights() {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<dt> distribution(mean_val, std_dev);

    weights = Tensor({ input_dim, output_dim }, {}, "Embedding");

    int n = weights.product(weights.shape);
    for (int i = 0; i < n; i++)
        weights.data[i] = distribution(generator);

    std::cout << "Initialized Embedding layer (" << input_dim << "x" << output_dim << ")\n";
}

void Embedding::forward(std::vector<int>& input_indices, Tensor& out, int B, int T) {
    out.gather(weights, input_indices, B, T);
}

void Embedding::update(dt lr) {
    size_t numel = weights.product(weights.shape);

    if (global_cuda_enabled) {
        int threads = 512;
        int blocks = (numel + threads - 1) / threads;

        sgd_step_kernel << <blocks, threads >> > (
            weights.d_data,
            weights.d_grad,
            lr,
            numel
            );
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

    }
    else {
        for (size_t i = 0; i < numel; i++) {
            weights.data[i] -= lr * weights.grad[i];
            weights.grad[i] = 0.0f;
        }
    }
}

Tensor Embedding::get_weights() const {
    return weights;
}

void Embedding::to_gpu() {
    weights.to_gpu();
    weights.constant_data = true;
}

void Embedding::to_cpu() {
    weights.to_cpu();
}





// -------- Linear --------

Linear::Linear(int input_dim_, int output_dim_, bool bias_, double std_dev_, double mean_val_)
    : input_dim(input_dim_), output_dim(output_dim_), bias(bias_), std_dev(std_dev_), mean_val(mean_val_) {
    initialize_weights();
}

Linear::Linear()
    : input_dim(0), output_dim(0), bias(false), std_dev(0.1), mean_val(0.0) {}

void Linear::initialize_weights() {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<dt> distribution(mean_val, std_dev);

    weights = Tensor({ output_dim, input_dim }, {}, "Linear weights");
    bias_v = Tensor(std::vector<int>{ (int)output_dim }, {}, "Linear bias");

    int w_numel = weights.product(weights.shape);
    for (int i = 0; i < w_numel; i++)
        weights.data[i] = distribution(generator);

    int b_numel = bias_v.product(bias_v.shape);
    for (int i = 0; i < b_numel; i++)
        bias_v.data[i] = distribution(generator);
}

void Linear::forward(Tensor& input, Tensor& output) {
    if (bias) {
        
        matmul_result = Tensor({}, "matmul_result");
        matmul_result.weighted_sum(input, weights);
        output.add(matmul_result, bias_v);
    }
    else {
        output.weighted_sum(input, weights);
    }
}

void Linear::update(dt lr) {
    size_t w_numel = weights.product(weights.shape);

    if (global_cuda_enabled) {
        int threads = 512;

        // --- Update weights ---
        int w_blocks = (w_numel + threads - 1) / threads;
        sgd_step_kernel << <w_blocks, threads >> > (weights.d_data, weights.d_grad, lr, w_numel);

        // --- Update bias (if used) ---
        if (bias) {
            size_t b_numel = bias_v.product(bias_v.shape);
            int b_blocks = (b_numel + threads - 1) / threads;
            sgd_step_kernel << <b_blocks, threads >> > (bias_v.d_data, bias_v.d_grad, lr, b_numel);
        }

        CHECK_CUDA(cudaGetLastError());
    }
    else {
        for (size_t i = 0; i < w_numel; i++) {
            weights.data[i] -= lr * weights.grad[i];
            weights.grad[i] = 0.0f;
        }

        if (bias) {
            size_t b_numel = bias_v.product(bias_v.shape);
            for (size_t i = 0; i < b_numel; i++) {
                bias_v.data[i] -= lr * bias_v.grad[i];
                bias_v.grad[i] = 0.0f;
            }
        }
    }
}


void Linear::to_gpu() {
    weights.to_gpu();
    if (bias) bias_v.to_gpu();
}

void Linear::to_cpu() {
    weights.to_cpu();
    if (bias) bias_v.to_cpu();
}

Tensor Linear::get_weights() const {
    return weights;
}

void Linear::set_weights(const std::vector<dt>& data) {
    weights.data = data;
}

Tensor Linear::get_bias() const {
    return bias_v;
}

void Linear::set_bias(const std::vector<dt>& data) {
    bias_v.data = data;
}

void Linear::zero_grad() {
    weights.zero_grad();
    bias_v.zero_grad();
}




// -------- Utility functions --------

std::string getWordBetweenDelimiters(const std::string& input, int startIndex, int endIndex) {
    std::vector<size_t> delimiter_positions;

    for (size_t i = 0; i < input.length(); i++) {
        if (input[i] == '.' || input[i] == ':') {
            delimiter_positions.push_back(i);
        }
    }

    if (startIndex >= (int)delimiter_positions.size() || endIndex >= (int)delimiter_positions.size()) {
        return "";
    }

    size_t startPos = delimiter_positions[startIndex] + 1;
    size_t endPos = delimiter_positions[endIndex];

    return input.substr(startPos, endPos - startPos);
}

int extract_number_between(const std::string& text, const std::string& start, const std::string& end) {
    size_t start_pos = text.find(start);
    if (start_pos == std::string::npos) return -1;

    start_pos += start.length();
    size_t end_pos = text.find(end, start_pos);
    if (end_pos == std::string::npos) return -1;

    std::string number_str = text.substr(start_pos, end_pos - start_pos);
    return std::stoi(number_str);
}

dt get_lr(const int& step, const Optimizer& opt) {
    if (step < opt.warmup_steps)
        return opt.max_lr * (step + 1) / opt.warmup_steps;

    if (step >= opt.max_steps)
        return opt.min_lr;

    dt decay_ratio = (dt)(step - opt.warmup_steps) / (opt.max_steps - opt.warmup_steps);
    assert(decay_ratio >= 0 && decay_ratio <= 1);

    dt coeff = 0.5 * (1.0 + std::cos(M_PI * decay_ratio));
    return opt.min_lr + coeff * (opt.max_lr - opt.min_lr);
}
