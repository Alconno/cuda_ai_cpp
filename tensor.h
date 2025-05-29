#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include "data_type.h"
#include "device_functions.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <functional>
#include <cmath>
#include <vector>
#include <set>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <queue>
#include <variant>
#include <memory>
#include <cassert>
#include <numeric>
#include <algorithm>

// Global CUDA flag
extern bool global_cuda_enabled;

// Helper print vector functions
template <typename T>
inline void pv(const std::vector<T>& vtr) {
    for (const auto& i : vtr) std::cout << i << " ";
    std::cout << "\n";
}

template <typename T>
inline void pv(const std::vector<std::vector<T>>& vtr) {
    for (const auto& i : vtr) pv(i);
    std::cout << "\n";
}

// CUDA error check helper
inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at " << file << ":" << line << " - "
            << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

namespace {
    // Basic ops
    static const auto op_add = [](dt x, dt y) { return x + y; };
    static const auto op_sub = [](dt x, dt y) { return x - y; };
    static const auto op_mul = [](dt x, dt y) { return x * y; };
    static const auto op_div = [](dt x, dt y) { return x / y; };

    // Gradients for ops
    static const auto grad_add_a = [](dt, dt) { return (dt)1.0; };
    static const auto grad_add_b = [](dt, dt) { return (dt)1.0; };
    static const auto grad_sub_a = [](dt, dt) { return (dt)1.0; };
    static const auto grad_sub_b = [](dt, dt) { return (dt)-1.0; };
    static const auto grad_mul_a = [](dt x, dt y) { return y; };
    static const auto grad_mul_b = [](dt x, dt y) { return x; };
    static const auto grad_div_a = [](dt, dt y) { return (dt)1.0 / y; };
    static const auto grad_div_b = [](dt x, dt y) { return -x / (y * y); };
}

class Tensor {
public:
    // Data storage
    std::vector<dt> data;
    std::vector<dt*> data_view;
    std::vector<dt> grad;
    std::vector<int> shape;
    std::vector<Tensor*> topo_order;

    // CUDA pointers
    dt* d_data = nullptr;
    dt* d_grad = nullptr;
    bool constant_data = false; // Data persistence flag

    // Constructors
    Tensor() {}

    Tensor(std::vector<int> shape, dt starting_val, std::string name = "", bool const_data = false, bool requires_grad = true)
        : shape(shape),
        data(product(shape), starting_val),
        grad(requires_grad ? product(shape) : 0, 0.0),
        name(name),
        requires_grad(requires_grad),
        constant_data(const_data)
    {}

    Tensor(dt scalar, std::string name = "", bool requires_grad = true)
        : shape({ 1 }),
        data({ scalar }),
        grad(requires_grad ? std::vector<dt>(1, 0.0) : std::vector<dt>{}),
        name(name),
        requires_grad(requires_grad)
    {}

    template <typename ShapeT, typename DataT>
    Tensor(std::vector<int>& shape_input, std::vector<dt>&& data_input,
        std::string name = "", bool const_data = false, bool requires_grad = true)
        : shape(std::forward<ShapeT>(shape_input)),
        data(std::forward<DataT>(data_input)),
        grad(requires_grad ? std::vector<dt>(product(shape), 0.0) : std::vector<dt>{}),
        name(name),
        requires_grad(requires_grad),
        constant_data(const_data)
    {
        grad = std::vector<dt>(product(shape), 0.0);
        int prod_shape = product(shape);
        while (data.size() < prod_shape)
            data.push_back((dt)0.0);

        if (data.size() != prod_shape) {
            std::cout << data.size() << std::endl;
            throw std::runtime_error("Data size does not match shape product.");
        }
    }

    ~Tensor() {
        if (d_data || d_grad) {
            std::cout << "Freeing out of scope: " << name << std::endl;
            free();
        }
    }

    // CUDA management
    void alloc_gpu();
    void to_cpu();
    void to_gpu();
    void free();
    void free_all();

    // Autograd
    std::unordered_set<Tensor*> prev;
    std::vector<std::function<void()>> backwardFuncs;
    bool requires_grad = true;
    std::string name = "";

    // Core utilities
    static int product(const std::vector<int>& shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    void resize_like(Tensor& other);
    void backward();
    void zero_grad();
    size_t numel() const { return product(shape); }

    // Tensor operations
    void add(Tensor& a, Tensor& b);
    void sub(Tensor& a, Tensor& b);
    void mul(Tensor& a, Tensor& b);
    void div(Tensor& a, Tensor& b);

    void apply_op(
        Tensor& a,
        Tensor& b,
        const std::function<dt(dt, dt)>& op,
        const std::function<dt(dt, dt)>& dOp_a,
        const std::function<dt(dt, dt)>& dOp_b,
        const char op_char);

    void weighted_sum(Tensor& a, Tensor& out);
    void softmax(Tensor& out);
    void cross_entropy(std::vector<int>& targets, Tensor& out);
    void gather(Tensor& weight, std::vector<int>& input_indices, const int& B, const int& T);
    void scaled_dot_product_attention(std::unordered_map<std::string, Tensor>& mats);
    void gelu(Tensor& out, bool approximate);
    void square(Tensor& a);
    void sqrt(Tensor& a);
    void mean(Tensor& input, bool keepdim);
    void sum(Tensor& out);
    void reduce(Tensor& out);
    void prod(Tensor& out);
    void divreduce(Tensor& out);

    // Shape operations
    void slice(int dim, int start, int end, std::shared_ptr<Tensor>& q);
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;

    // Computational graph
    static std::vector<Tensor*> topological_sort(Tensor* root);

    // Debug/IO
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
};


std::vector<int> broadcast_shapes(const std::vector<int>& shape1, const std::vector<int>& shape2);

std::vector<int> unravel_index(int flat_index, const std::vector<int>& shape);

int ravel_index(const std::vector<int>& indices, const std::vector<int>& shape);

int broadcast_index(const std::vector<int>& indices,
    const std::vector<int>& orig_shape,
    const std::vector<int>& result_shape);

std::vector<int> compute_strides(const std::vector<int>& shape);

void print_gpu_data(const float* d_ptr, size_t size, const std::string& label = "");

#endif // TENSOR_H