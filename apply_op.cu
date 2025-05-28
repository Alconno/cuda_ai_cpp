#include "tensor.h"
#include "kernels.h"

void Tensor::apply_op(
    Tensor& a, Tensor& b,
    const std::function<dt(dt, dt)>& op,
    const std::function<dt(dt, dt)>& dOp_a,
    const std::function<dt(dt, dt)>& dOp_b,
    char op_char
) {
    // 1. Broadcast shapes and initialize result tensor
    this->shape = broadcast_shapes(a.shape, b.shape);
    int numel = product(this->shape);

    this->data.assign(numel, 0.0);
    this->grad.assign(numel, 0.0);
    this->requires_grad = a.requires_grad || b.requires_grad;

    int ndim = this->shape.size();
    int a_offset = ndim - a.shape.size();
    int b_offset = ndim - b.shape.size();

    std::vector<int> a_indices;
    std::vector<int> b_indices;

    // 2. Extract broadcast dimensions for GPU
    int B = (ndim >= 4) ? this->shape[ndim - 4] : 1;
    int H = (ndim >= 3) ? this->shape[ndim - 3] : 1;
    int T = (ndim >= 2) ? this->shape[ndim - 2] : 1;
    int J = (ndim >= 1) ? this->shape[ndim - 1] : 1;

    auto get_dim = [](const std::vector<int>& shape, int idx_from_end) {
        int offset = shape.size();
        return (offset >= idx_from_end) ? shape[offset - idx_from_end] : 1;
        };

    int a_H = get_dim(a.shape, 3);
    int a_T = get_dim(a.shape, 2);
    int a_J = get_dim(a.shape, 1);

    int b_H = get_dim(b.shape, 3);
    int b_T = get_dim(b.shape, 2);
    int b_J = get_dim(b.shape, 1);

    // 3. Forward Pass
    if (global_cuda_enabled) {
        this->alloc_gpu();
        dt* d_a = a.d_data;
        dt* d_b = b.d_data;
        dt* d_out = this->d_data;

        assert(d_a && d_b && d_out);

        dim3 block(32, 16);
        dim3 grid((J + block.x - 1) / block.x, (T + block.y - 1) / block.y, B * H);

        broadcast_binary_kernel << <grid, block >> > (
            d_a, d_b, d_out,
            B, H, T, J,
            a_H, a_T, a_J,
            b_H, b_T, b_J,
            op_char
            );
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

    }
    else {
        a_indices.resize(numel, 0.0);
        b_indices.resize(numel, 0.0);
#pragma omp parallel for
        for (int i = 0; i < numel; ++i) {
            int rem = i;
            int stride_a = 1, stride_b = 1;

            for (int d = ndim - 1; d >= 0; --d) {
                int dim = this->shape[d];
                int idx = rem % dim;
                rem /= dim;

                int a_dim = d - a_offset;
                if (a_dim >= 0) {
                    int a_size = a.shape[a_dim];
                    a_indices[i] += ((a_size == 1) ? 0 : idx) * stride_a;
                    stride_a *= a_size;
                }

                int b_dim = d - b_offset;
                if (b_dim >= 0) {
                    int b_size = b.shape[b_dim];
                    b_indices[i] += ((b_size == 1) ? 0 : idx) * stride_b;
                    stride_b *= b_size;
                }
            }

            this->data[i] = op(a.data[a_indices[i]], b.data[b_indices[i]]);
        }
    }

    // 4. Backward Pass
    if (this->requires_grad) {
        this->prev.insert(&a);
        this->prev.insert(&b);

        a.grad.resize(a.data.size(), 0.0);
        b.grad.resize(b.data.size(), 0.0);

        if (global_cuda_enabled) {
            this->backwardFuncs.push_back([=, &a, &b]() {
                dt* d_a = a.d_data;
                dt* d_b = b.d_data;
                dt* d_out_grad = this->d_grad;
                dt* d_a_grad = a.d_grad;
                dt* d_b_grad = b.d_grad;

                dim3 block(32, 16);
                dim3 grid((J + block.x - 1) / block.x, (T + block.y - 1) / block.y, B * H);

                broadcast_binary_kernel_backward << <grid, block >> > (
                    d_a, d_b, d_a_grad, d_b_grad, d_out_grad,
                    B, H, T, J, a_H, a_T, a_J, b_H, b_T, b_J,
                    op_char
                    );
                cudaDeviceSynchronize();
                CHECK_CUDA(cudaGetLastError());
                });

        }
        else {
            this->backwardFuncs.push_back([this, a_indices, b_indices, &a, &b, numel, dOp_a, dOp_b]() {
                for (int i = 0; i < numel; ++i) {
                    if (a.requires_grad)
                        a.grad[a_indices[i]] += this->grad[i] * dOp_a(a.data[a_indices[i]], b.data[b_indices[i]]);
                    if (b.requires_grad)
                        b.grad[b_indices[i]] += this->grad[i] * dOp_b(a.data[a_indices[i]], b.data[b_indices[i]]);
                }
                });
        }
    }
}


void Tensor::add(Tensor& a, Tensor& b) {
    apply_op(a, b, op_add, grad_add_a, grad_add_b, '+');
}
void Tensor::sub(Tensor& a, Tensor& b) {
    apply_op(a, b, op_sub, grad_sub_a, grad_sub_b, '-');
}
void Tensor::mul(Tensor& a, Tensor& b) {
    apply_op(a, b, op_mul, grad_mul_a, grad_mul_b, '*');
}
void Tensor::div(Tensor& a, Tensor& b) {
    apply_op(a, b, op_div, grad_div_a, grad_div_b, '/');
}