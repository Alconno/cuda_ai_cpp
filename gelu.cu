#include "tensor.h"
#include "kernels.h"

void Tensor::gelu(Tensor& out, bool approximate) {
    // Setup output tensor
    out.requires_grad = this->requires_grad;
    out.data.resize(this->numel());
    out.grad.resize(this->numel());
    std::fill(out.data.begin(), out.data.end(), 0.0f);
    std::fill(out.grad.begin(), out.grad.end(), 0.0f);
    out.shape = this->shape;

    if (global_cuda_enabled) {
        // CUDA path
        out.alloc_gpu();

        dt* d_x = this->d_data;
        dt* d_out = out.d_data;
        int N = (int)this->numel();
        dim3 block(512);
        dim3 grid((N + block.x - 1) / block.x);

        gelu_forward_kernel << <grid, block >> > (d_x, d_out, N, approximate);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        if (this->requires_grad) {
            out.prev.insert(this);
            out.backwardFuncs.push_back([this, out_ptr = &out, approximate]() {
                dt* d_grad_x = this->d_grad;
                dt* d_grad_out = out_ptr->d_grad;
                dt* d_x = this->d_data;
                int N = (int)this->numel();
                dim3 block(512);
                dim3 grid((N + block.x - 1) / block.x);

                gelu_backward_kernel << <grid, block >> > (d_x, d_grad_out, d_grad_x, N, approximate);
                cudaDeviceSynchronize();
                CHECK_CUDA(cudaGetLastError());
                });
        }
    }
    else {
        // CPU path helpers for backward
        auto backward_approximate = [this, &out]() {
            const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
            for (int i = 0; i < this->numel(); ++i) {
                float x = this->data[i];
                float x2 = x * x;
                float x3 = x2 * x;
                float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x3);
                float tanh_val = std::tanh(tanh_arg);
                float sech2 = 1.0f - tanh_val * tanh_val;

                float coeff = 0.5f * x * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x2);
                float dgelu_dx = 0.5f * (1.0f + tanh_val) + coeff;

                this->grad[i] += out.grad[i] * dgelu_dx;
            }
            };

        auto backward_exact = [this, &out]() {
            const float sqrt1_2 = 1.0f / std::sqrt(2.0f);
            const float sqrt2_over_pi = std::sqrt(2.0f / M_PI);
            for (int i = 0; i < this->numel(); ++i) {
                float x = this->data[i];
                float erf_term = std::erf(x * sqrt1_2);
                float exp_term = std::exp(-0.5f * x * x);
                float dgelu_dx = 0.5f * (1.0f + erf_term) + 0.5f * x * sqrt2_over_pi * exp_term;

                this->grad[i] += out.grad[i] * dgelu_dx;
            }
            };

        if (approximate) {
            // GELU approximate forward
            const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
            for (int i = 0; i < this->numel(); ++i) {
                float x = this->data[i];
                float x3 = x * x * x;
                float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x3);
                float tanh_val = std::tanh(tanh_arg);
                out.data[i] = 0.5f * x * (1.0f + tanh_val);
            }

            if (this->requires_grad) {
                out.prev.insert(this);
                out.backwardFuncs.push_back(backward_approximate);
            }
        }
        else {
            // GELU exact forward
            const float sqrt1_2 = 1.0f / std::sqrt(2.0f);
            for (int i = 0; i < this->numel(); ++i) {
                float x = this->data[i];
                out.data[i] = 0.5f * x * (1.0f + std::erf(x * sqrt1_2));
            }

            if (this->requires_grad) {
                out.prev.insert(this);
                out.backwardFuncs.push_back(backward_exact);
            }
        }
    }
}