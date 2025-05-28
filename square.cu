#include "tensor.h"
#include "kernels.h"

void Tensor::square(Tensor& a) {
    this->resize_like(a);

    if (global_cuda_enabled) {
        // GPU square
        this->alloc_gpu();

        dt* d_a_data = a.d_data;
        dt* d_out_data = this->d_data;

        int threads = 512;
        int blocks = (a.numel() + threads - 1) / threads;

        square_kernel << <blocks, threads >> > (
            d_a_data,
            d_out_data
            );

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        // Optional: copy result to host
        // cudaMemcpy(this->data.data(), d_out_data, this->data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
    }
    else {
        // CPU square
        for (int i = 0; i < a.numel(); ++i) {
            this->data[i] = a.data[i] * a.data[i];
        }
    }

    if (a.requires_grad) {
        this->requires_grad = true;
        this->prev.insert(&a);

        if (global_cuda_enabled) {
            this->backwardFuncs.push_back([this, &a]() {
                dt* d_a_data = a.d_data;
                dt* d_a_grad = a.d_grad;
                dt* d_out_grad = this->d_grad;

                int threads = 512;
                int blocks = (a.numel() + threads - 1) / threads;

                square_backward_kernel << <blocks, threads >> > (
                    d_a_data,
                    d_a_grad,
                    d_out_grad
                    );

                cudaDeviceSynchronize();
                CHECK_CUDA(cudaGetLastError());

                // Optional: copy grad to host
                // cudaMemcpy(a.grad.data(), d_a_grad, a.grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);
                });
        }
        else {
            this->backwardFuncs.push_back([this, &a]() {
                for (int i = 0; i < a.numel(); ++i) {
                    a.grad[i] += this->grad[i] * 2.0 * a.data[i];
                }
                });
        }
    }
}