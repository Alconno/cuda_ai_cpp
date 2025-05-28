#include "tensor.h"
#include "kernels.h"

void Tensor::sum(Tensor& out) {
    out.data.resize({ 1 });
    out.requires_grad = this->requires_grad;

    if (global_cuda_enabled) {
        out.alloc_gpu();

        dt* d_input = this->d_data;
        dt* d_output = out.d_data;
        int size = this->numel();

        int threads = 512;
        int blocks = (size + threads - 1) / threads;

        // sum_kernel<<<blocks, threads>>>(d_input, d_output, size);

        // cudaMemcpy(out.data.data(), d_output, sizeof(dt), cudaMemcpyDeviceToHost);
    }
    else {
        dt sum = 0;
#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < this->data.size(); ++i) {
            sum += this->data[i];
        }
        out.data[0] = sum;
    }

    if (this->requires_grad) {
        out.prev.insert(this);

        if (global_cuda_enabled) {
            out.backwardFuncs.push_back([this, &out]() {
                dt* d_grad_input = this->d_grad;
                dt grad_output = out.grad[0];

                dt* d_grad_output;
                cudaMalloc(&d_grad_output, sizeof(dt));
                cudaMemcpy(d_grad_output, &grad_output, sizeof(dt), cudaMemcpyHostToDevice);

                int threads = 512;
                int blocks = (this->numel() + threads - 1) / threads;

                // sum_backward_kernel<<<blocks, threads>>>(d_grad_input, *d_grad_output, this->numel());

                // cudaMemcpy(this->grad.data(), d_grad_input, this->grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);
                cudaFree(d_grad_output);
                });
        }
        else {
            out.backwardFuncs.push_back([this, &out]() {
                dt g = out.grad[0];
#pragma omp parallel for
                for (int i = 0; i < this->numel(); ++i) {
                    this->grad[i] += g;
                }
                });
        }
    }
}


void Tensor::reduce(Tensor& out) {
    out.requires_grad = this->requires_grad;

    out.data[0] = this->data[0];
    for (int i = 1; i < this->numel(); ++i) {
        out.data[0] -= this->data[i];
    }

    // Backward function
    if (this->requires_grad) {
        out.backwardFuncs.push_back([this, &out]() {
            this->grad[0] += out.grad[0];
            for (int i = 1; i < this->numel(); ++i) {
                this->grad[i] -= out.grad[0];
            }
            });
    }
}

void Tensor::prod(Tensor& out) {
    out.requires_grad = this->requires_grad;

    out.data[0] = 1.0;
    for (int i = 0; i < this->numel(); ++i) {
        out.data[0] *= this->data[i];
    }

    if (this->requires_grad) {
        out.backwardFuncs.push_back([this, &out]() {
            for (int i = 0; i < this->numel(); ++i) {
                dt partial = out.data[0] / this->data[i]; // ∂(prod)/∂xᵢ = prod / xᵢ
                this->grad[i] += out.grad[0] * partial;
            }
            });
    }
}

void Tensor::divreduce(Tensor& out) {
    out.requires_grad = this->requires_grad;

    out.data[0] = this->data[0];
    for (int i = 1; i < this->numel(); ++i) {
        out.data[0] /= this->data[i];
    }

    if (this->requires_grad) {
        out.backwardFuncs.push_back([this, &out]() {
            dt denom = 1.0;
            for (int i = 1; i < this->numel(); ++i)
                denom *= this->data[i];
            this->grad[0] += out.grad[0] * (1.0 / denom);

            // ∂out/∂xᵢ for i > 0
            for (int i = 1; i < this->numel(); ++i) {
                this->grad[i] -= out.grad[0] * (out.data[0] / this->data[i]);
            }
            });
    }
}