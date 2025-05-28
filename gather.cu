#include "tensor.h"
#include "kernels.h"

void Tensor::gather(Tensor& weight, std::vector<int>& input_indices, const int& B, const int& T) {
    int C = weight.shape[1];

    this->shape = { B, T, C };
    this->data.resize(B * T * C, 0.0f);
    this->grad.resize(B * T * C, 0.0f);

    if (global_cuda_enabled) {
        this->alloc_gpu();

        dt* d_weight = weight.d_data;
        dt* d_out = this->d_data;

        int* d_indices;
        cudaMalloc(&d_indices, input_indices.size() * sizeof(int));
        cudaMemcpy(d_indices, input_indices.data(), input_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

        dim3 block(32, 32);
        dim3 grid((C + block.x - 1) / block.x, (B * T + block.y - 1) / block.y);

        gather_kernel << <grid, block >> > (d_weight, d_out, d_indices, B * T, C);

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());
        cudaFree(d_indices);
    }
    else {
        // CPU gather
#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < C; ++c)
                    this->data[((b * T + t) * C) + c] = weight.data[(input_indices[b * T + t] * C) + c];
    }

    // Backward
    if (weight.requires_grad) {
        this->requires_grad = true;
        this->prev.insert(&weight);

        if (global_cuda_enabled) {
            this->backwardFuncs.push_back([this, &input_indices, &weight, B, T]() {
                int C = weight.shape[1];

                dt* d_out_grad = this->d_grad;
                dt* d_weight_grad = weight.d_grad;

                int* d_indices;
                cudaMalloc(&d_indices, input_indices.size() * sizeof(int));
                cudaMemcpy(d_indices, input_indices.data(), input_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

                dim3 block(std::min(C, 256));
                dim3 grid(B * T);

                gather_backward_kernel << <grid, block >> > (d_out_grad, d_indices, d_weight_grad, B * T, C);

                cudaDeviceSynchronize();
                CHECK_CUDA(cudaGetLastError());
                cudaFree(d_indices);
                });
        }
        else {
            this->backwardFuncs.push_back([this, &input_indices, &weight, &B, &T]() {
                int C = weight.shape[1];
#pragma omp parallel for collapse(2)
                for (int b = 0; b < B; ++b)
                    for (int t = 0; t < T; ++t)
                        for (int c = 0; c < C; ++c)
                            weight.grad[(input_indices[b * T + t] * C) + c] += this->grad[((b * T + t) * C) + c];
                });
        }
    }
}