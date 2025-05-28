#include "tensor.h"
#include "kernels.h"


void Tensor::softmax(Tensor& out) {
    assert(this->shape.size() == 3);
    int B = this->shape[0];
    int T = this->shape[1];
    int V = this->shape[2];

    out.shape = this->shape;
    out.data.resize(this->numel());
    out.grad.resize(this->numel(), 0.0);
    out.requires_grad = this->requires_grad;

    if (global_cuda_enabled) {
        out.alloc_gpu();

        dt* d_in = this->d_data;
        dt* d_out = out.d_data;

        dim3 block(16, 32);
        dim3 grid(
            (B + block.x - 1) / block.x,
            (T + block.y - 1) / block.y
        );

        softmax_forward_kernel << <grid, block >> > (d_in, d_out, B, T, V);

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        //cudaMemcpy(out.data.data(), d_out, out.data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
    }
    else {
        // CPU forward with OpenMP parallelization
#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                int base = (b * T + t) * V;

                // Find max for numerical stability
                dt max_val = this->data[base];
                for (int v = 1; v < V; ++v) {
                    max_val = std::max(max_val, this->data[base + v]);
                }

                dt sum = 0.0;
                for (int v = 0; v < V; ++v) {
                    out.data[base + v] = std::exp(this->data[base + v] - max_val);
                    sum += out.data[base + v];
                }

                dt inv_sum = 1.0 / sum;
                for (int v = 0; v < V; ++v) {
                    out.data[base + v] *= inv_sum;
                }
            }
        }
    }

    if (this->requires_grad) {
        out.prev.insert(this);

        if (global_cuda_enabled) {
            out.backwardFuncs.push_back([this, &out, B, T, V]() {
                dt* d_out = out.d_data;
                dt* d_grad_in = this->d_grad;
                dt* d_grad_out = out.d_grad;

                dim3 block(16, 32);
                dim3 grid(
                    (B + block.x - 1) / block.x,
                    (T + block.y - 1) / block.y
                );

                softmax_backward_kernel << <grid, block >> > (d_out, d_grad_out, d_grad_in, B, T, V);

                cudaDeviceSynchronize();
                CHECK_CUDA(cudaGetLastError());

                //cudaMemcpy(this->grad.data(), d_grad_in, this->grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);
                });
        }
        else {
            out.backwardFuncs.push_back([this, &out, B, T, V]() {
#pragma omp parallel for collapse(2)
                for (int b = 0; b < B; ++b) {
                    for (int t = 0; t < T; ++t) {
                        int base = (b * T + t) * V;

                        dt* y = &out.data[base];
                        dt* dy = &out.grad[base];
                        dt* dx = &this->grad[base];

                        // Jacobian-vector product for softmax gradient
                        dt dot = 0.0;
                        for (int j = 0; j < V; ++j) {
                            dot += y[j] * dy[j];
                        }

                        for (int i = 0; i < V; ++i) {
                            dx[i] += y[i] * (dy[i] - dot);
                        }
                    }
                }
                });
        }
    }
}

