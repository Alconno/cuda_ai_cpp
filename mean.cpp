#include "tensor.h"
#include "kernels.h"

void Tensor::mean(Tensor& input, bool keepdim) {
    int last_dim = (int)input.shape.size() - 1;
    int reduce_size = input.shape[last_dim];

    // Determine output shape
    std::vector<int> out_shape = input.shape;
    if (keepdim) {
        out_shape[last_dim] = 1;
    }
    else {
        out_shape.pop_back();
    }

    // Allocate and initialize this tensor
    this->shape = out_shape;
    int out_numel = Tensor::product(out_shape);
    this->data.assign(out_numel, 0.0f);
    this->grad.assign(out_numel, 0.0f);
    this->requires_grad = input.requires_grad;

    std::vector<int> input_strides = compute_strides(input.shape);
    std::vector<int> out_strides = compute_strides(out_shape);

    if (global_cuda_enabled) {
        // GPU mode
        this->alloc_gpu();

        int input_shape_size = input.shape.size();
        int out_shape_size = out_shape.size();

        // Allocate and copy stride arrays to device
        int* d_input_strides, * d_out_strides;
        cudaMalloc(&d_input_strides, sizeof(int) * input_shape_size);
        cudaMalloc(&d_out_strides, sizeof(int) * out_shape_size);

        cudaMemcpy(d_input_strides, input_strides.data(), sizeof(int) * input_shape_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_strides, out_strides.data(), sizeof(int) * out_shape_size, cudaMemcpyHostToDevice);

        int threads = 512;
        int blocks = (input.numel() + threads - 1) / threads;

        mean_kernel << <blocks, threads >> > (
            input.d_data, this->d_data,
            input.shape.data(), out_shape.data(),
            input_strides.data(), out_strides.data(),
            input_shape_size, out_shape_size,
            reduce_size, input.numel(),
            keepdim,
            d_input_strides, d_out_strides
            );

        cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());
        cudaFree(d_input_strides);
        cudaFree(d_out_strides);
    }
    else {
        // CPU mode
        for (int i = 0; i < input.numel(); ++i) {
            int idx[10], out_idx[10];
            int flat = i;

            for (int d = 0; d < input.shape.size(); ++d) {
                idx[d] = flat / input_strides[d];
                flat %= input_strides[d];
            }

            for (int d = 0; d < out_shape.size(); ++d) {
                out_idx[d] = idx[d];
            }
            if (keepdim) out_idx[last_dim] = 0;

            int j = 0;
            for (int d = 0; d < out_shape.size(); ++d) {
                j += out_idx[d] * out_strides[d];
            }

            this->data[j] += input.data[i] / (dt)reduce_size;
        }
    }

    // Backward pass setup
    if (input.requires_grad) {
        this->prev.insert(&input);

        if (global_cuda_enabled) {
            this->backwardFuncs.push_back([this, &input, reduce_size, keepdim]() {
                int input_shape_size = input.shape.size();
                int out_shape_size = this->shape.size();
                int numel = input.numel();

                dt* d_input_grad = input.d_grad;
                dt* d_out_grad = this->d_grad;

                int* d_input_shape, * d_out_shape, * d_input_strides, * d_out_strides;

                cudaMalloc(&d_input_shape, sizeof(int) * input_shape_size);
                cudaMalloc(&d_out_shape, sizeof(int) * out_shape_size);
                cudaMalloc(&d_input_strides, sizeof(int) * input_shape_size);
                cudaMalloc(&d_out_strides, sizeof(int) * out_shape_size);

                auto input_strides = compute_strides(input.shape);
                auto out_strides = compute_strides(this->shape);

                cudaMemcpy(d_input_shape, input.shape.data(), sizeof(int) * input_shape_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_out_shape, this->shape.data(), sizeof(int) * out_shape_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_input_strides, input_strides.data(), sizeof(int) * input_shape_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_out_strides, out_strides.data(), sizeof(int) * out_shape_size, cudaMemcpyHostToDevice);

                int threads = 512;
                int blocks = (numel + threads - 1) / threads;

                mean_backward_kernel << <blocks, threads >> > (
                    d_input_grad, d_out_grad,
                    d_input_shape, d_out_shape,
                    d_input_strides, d_out_strides,
                    input_shape_size, out_shape_size,
                    reduce_size, numel, (int)keepdim
                    );

                cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());

                cudaFree(d_input_shape);
                cudaFree(d_out_shape);
                cudaFree(d_input_strides);
                cudaFree(d_out_strides);
                });

        }
        else {
            this->backwardFuncs.push_back([this, &input, reduce_size, keepdim]() {
                auto input_strides = compute_strides(input.shape);
                auto out_strides = compute_strides(this->shape);
                int last_dim = (int)input.shape.size() - 1;

                for (int i = 0; i < input.numel(); ++i) {
                    std::vector<int> idx(input.shape.size());
                    int flat = i;

                    for (int d = 0; d < input.shape.size(); ++d) {
                        idx[d] = flat / input_strides[d];
                        flat %= input_strides[d];
                    }

                    std::vector<int> out_idx = idx;
                    if (keepdim) {
                        out_idx[last_dim] = 0;
                    }
                    else {
                        out_idx.pop_back();
                    }

                    int j = 0;
                    for (int d = 0; d < out_idx.size(); ++d) {
                        j += out_idx[d] * out_strides[d];
                    }

                    input.grad[i] += this->grad[j] / (dt)reduce_size;
                }
                });
        }
    }
}