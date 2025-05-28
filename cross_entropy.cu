#include "tensor.h"
#include "kernels.h"

void Tensor::cross_entropy(std::vector<int>& targets, Tensor& out) {
    assert(this->shape.size() == 3);
    int B = this->shape[0];
    int T = this->shape[1];
    int V = this->shape[2];

    assert(targets.size() == B * T);
    assert(this->data.size() == B * T * V);

    out.shape = { 1 };
    out.data.resize(1, 0.0);
    out.grad.resize(1, 0.0);
    out.requires_grad = this->requires_grad;

    // Forward pass
    if (global_cuda_enabled) {
        out.alloc_gpu();

        dt* d_in = this->d_data;
        dt* d_out = out.d_data;

        int* d_targets;
        cudaMalloc(&d_targets, targets.size() * sizeof(int));
        cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(int), cudaMemcpyHostToDevice);

        dim3 block(16, 32);
        dim3 grid(
            (B + block.x - 1) / block.x,
            (T + block.y - 1) / block.y
        );

        cross_entropy_kernel << <grid, block >> > (d_in, d_out, d_targets, B, T, V);

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        cudaMemcpy(out.data.data(), d_out, out.data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
        cudaFree(d_targets);

        out.data[0] /= (B * T);
    }
    else {
        dt loss_sum = 0.0;
        for (int i = 0; i < B * T; ++i) {
            int target_idx = targets[i];
            assert(target_idx >= 0 && target_idx < V);
            int idx = i * V + target_idx;
            dt prob = this->data[idx];
            loss_sum += -std::log(prob + 1e-10);
        }
        out.data[0] = loss_sum / (B * T);
    }

    // Backward pass
    if (this->requires_grad) {
        out.prev.insert(this);

        if (global_cuda_enabled) {
            out.backwardFuncs.push_back([this, &targets, &out, B, T, V]() {
                dim3 block(16, 32);
                dim3 grid(
                    (B + block.x - 1) / block.x,
                    (T + block.y - 1) / block.y
                );

                int* d_targets;
                cudaMalloc(&d_targets, targets.size() * sizeof(int));
                cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(int), cudaMemcpyHostToDevice);

                cross_entropy_backward_kernel << <grid, block >> > (
                    this->d_data,
                    this->d_grad,
                    d_targets,
                    B, T, V,
                    out.grad[0]
                    );

                cudaDeviceSynchronize();
                CHECK_CUDA(cudaGetLastError());

                cudaFree(d_targets);
                });
        }
        else {
            out.backwardFuncs.push_back([this, &targets, &out, B, T, V]() {
                for (int i = 0; i < B * T; ++i) {
                    int target_idx = targets[i];
                    int idx = i * V + target_idx;
                    dt prob = this->data[idx];
                    dt grad = (-1.0 / (prob + 1e-10)) * (out.grad[0] / (B * T));
                    this->grad[idx] += grad;
                }
                });
        }
    }
}