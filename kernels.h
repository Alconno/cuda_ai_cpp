#pragma once
#include <omp.h>
#include "device_functions.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_type.h"

#define M_PI 3.14159265358979323846

__global__ void weighted_sum_kernel(
    const dt* __restrict__ a_data,
    const dt* __restrict__ b_data,
    dt* __restrict__ out_data,
    int B, int H, int T, int J, int C, int b_rows);

__global__ void weighted_sum_backward_kernel(
    const dt* __restrict__ a_data,
    const dt* __restrict__ b_data,
    dt* __restrict__ a_grad,
    dt* __restrict__ b_grad,
    const dt* __restrict__ out_grad,
    int B, int H, int T, int J, int C, int b_rows);

__global__ void broadcast_binary_kernel(
    const dt* __restrict__ a_data,
    const dt* __restrict__ b_data,
    dt* __restrict__ out_data,
    int B, int H, int T, int J,
    int a_H, int a_T, int a_J,
    int b_H, int b_T, int b_J,
    char op
);

__global__ void broadcast_binary_kernel_backward(
    const dt* __restrict__ a_data,
    const dt* __restrict__ b_data,
    dt* __restrict__ a_grad,
    dt* __restrict__ b_grad,
    const dt* __restrict__ out_grad,
    int B, int H, int T, int J,
    int a_H, int a_T, int a_J,
    int b_H, int b_T, int b_J,
    char op
);

__global__ void mean_kernel(
    const dt* __restrict__ input_data,
    dt* __restrict__ out_data,
    const int* __restrict__ input_shape,
    const int* __restrict__ out_shape,
    const int* __restrict__ input_strides,
    const int* __restrict__ out_strides,
    int input_shape_size,
    int out_shape_size,
    int reduce_size,
    int numel,
    int keepdim,
    const int* __restrict__ d_input_strides,
    const int* __restrict__ d_out_strides
);

__global__ void mean_backward_kernel(
    dt* __restrict__ input_grad,
    const dt* __restrict__ out_grad,
    const int* __restrict__ input_shape,
    const int* __restrict__ out_shape,
    const int* __restrict__ input_strides,
    const int* __restrict__ out_strides,
    int input_shape_size,
    int out_shape_size,
    int reduce_size,
    int numel,
    int keepdim
);

__global__ void square_kernel(const dt* __restrict__ input_data, dt* __restrict__ out_data);
__global__ void square_backward_kernel(
    const dt* __restrict__ input_data,
    dt* __restrict__ input_grad,
    const dt* __restrict__ out_grad);

__global__ void sqrt_kernel(const dt* in, dt* out, const int N);
__global__ void sqrt_backward_kernel(const dt* in, const dt* grad_out, dt* grad_in, const int N);


__global__ void gather_kernel(
    const dt* __restrict__ weight, dt* __restrict__ out,
    const int* indices, int BT, int C);
__global__ void gather_backward_kernel(
    const dt* __restrict__ out_grad,
    const int* __restrict__ indices,
    dt* __restrict__ weight_grad,
    int BT, int C);

#if 0
__global__ void sum_kernel(const dt* __restrict__ input, dt* __restrict__ output, int size);
__global__ void sum_backward_kernel(dt* __restrict__ grad_input, dt grad_output, int size);
#endif

__global__ void gelu_forward_kernel(const float* x, float* out, int N, bool approximate);
__global__ void gelu_backward_kernel(const float* x, const float* out_grad, float* x_grad, int N, bool approximate);

// Attention
__global__ void scaled_dot_qk_softmax_kernel(
    const dt* __restrict__ qkv,
    dt* __restrict__ scores,
    dt* __restrict__ exps,
    dt* __restrict__ smax,
    int B, int H, int T, int C, int head_dim
);

__global__ void smax_weighted_v_kernel(
    const dt* __restrict__ qkv,
    const dt* __restrict__ smax,
    dt* __restrict__ v_out,
    int B, int H, int T, int C, int head_dim
);

__global__ void backward_attention_kernel(
    const dt* __restrict__ out_grad,
    const dt* __restrict__ smax,
    const dt* __restrict__ qkv,

    dt* __restrict__ smax_grad,
    dt* __restrict__ qkv_grad,
    dt* __restrict__ score_grad,

    int B, int n_head, int T, int C, int head_dim
);


__global__ void softmax_forward_kernel(
    const dt* __restrict__ input,
    dt* __restrict__ out,
    int B, int T, int V
);
__global__ void softmax_backward_kernel(
    const dt* __restrict__ out,
    const dt* __restrict__ out_grad,
    dt* __restrict__ input_grad,
    int B, int T, int V
);

__global__ void cross_entropy_kernel(
    const dt* __restrict__ input,
    dt* __restrict__ out,
    const int* __restrict__ targets,
    const int B, const int T, const int V
);
__global__ void cross_entropy_backward_kernel(
    const dt* __restrict__ probs,
    dt* __restrict__ grad_out,
    const int* __restrict__ targets,
    const int B, const int T, const int V,
    const dt grad_loss
);

__global__ void sgd_step_kernel(dt* weights, dt* grads, float lr, int numel);