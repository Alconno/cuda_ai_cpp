#include "kernels.h"
#include <stdio.h>
#include <iostream>

__inline__ __device__
float warp_reduce_sum_same_key(float val, int key, int lane_id) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        int other_key = __shfl_down_sync(0xffffffff, key, offset);
        float other_val = __shfl_down_sync(0xffffffff, val, offset);

        if (key == other_key)
            val += other_val;
    }
    return val;
}



__global__ void weighted_sum_kernel(
    const dt* __restrict__ a_data,
    const dt* __restrict__ b_data,
    dt* __restrict__ out_data,
    int B, int H, int T, int J, int C, int b_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z;

    if (i >= T || bh >= B * H) return;

    int b = bh / H;
    int h = bh % H;
    int i_idx = (bh * T + i) * C;


    int jj = (bh * J + j) % b_rows;
    int j_idx = jj * C;

    dt sum = 0.0;

#pragma unroll
    for (int k = 0; k < C; k++) {
        sum += a_data[i_idx + k] * b_data[j_idx + k];
    }

    out_data[((b * H + h) * T + i) * J + j] = sum;
    
}




__global__ void weighted_sum_backward_kernel(
    const dt* __restrict__ a_data,
    const dt* __restrict__ b_data,
    dt* __restrict__ a_grad,
    dt* __restrict__ b_grad,
    const dt* __restrict__ out_grad,
    int B, int H, int T, int J, int C, int b_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z;

    if (j >= J || i >= T || bh >= B * H) return;

    int b = bh / H;
    int h = bh % H;
    int a_idx = (bh * T + i) * C;

    int jj = (bh * J + j) % b_rows;
    int b_idx = jj * C;
    dt d_out = out_grad[((b * H + h) * T + i) * J + j];

#pragma unroll
    for (int k = 0; k < C; k++) {
        atomicAdd(&a_grad[a_idx + k], d_out * b_data[b_idx + k]);
        atomicAdd(&b_grad[b_idx + k], d_out * a_data[a_idx + k]);
    }

}





__global__ void broadcast_binary_kernel(
    const dt* __restrict__ a_data,
    const dt* __restrict__ b_data,
    dt* __restrict__ out_data,
    int B, int H, int T, int J,
    int a_H, int a_T, int a_J,
    int b_H, int b_T, int b_J,
    char op
)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int bh = blockIdx.z;

    if (j >= J || i >= T || bh >= B * H) return;

    int b = bh / H;
    int h = bh % H;

    int out_idx = ((b * H + h) * T + i) * J + j;

    int a_idx = ((b * a_H + (a_H == 1 ? 0 : h)) * a_T + (a_T == 1 ? 0 : i)) * a_J + (a_J == 1 ? 0 : j);
    int b_idx = ((b * b_H + (b_H == 1 ? 0 : h)) * b_T + (b_T == 1 ? 0 : i)) * b_J + (b_J == 1 ? 0 : j);

    dt a_val = a_data[a_idx];
    dt b_val = b_data[b_idx];

    
    if (op == '+') out_data[out_idx] = a_val + b_val;
    else if (op == '-') out_data[out_idx] = a_val - b_val; 
    else if (op == '*') out_data[out_idx] = a_val * b_val;
    else if (op == '/') out_data[out_idx] = a_val / b_val;
}

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
) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int bh = blockIdx.z;

    if (j >= J || i >= T || bh >= B * H) return;

    int b = bh / H;
    int h = bh % H;

    int out_idx = ((b * H + h) * T + i) * J + j;
    int a_idx = ((b * a_H + (a_H == 1 ? 0 : h)) * a_T + (a_T == 1 ? 0 : i)) * a_J + (a_J == 1 ? 0 : j);
    int b_idx = ((b * b_H + (b_H == 1 ? 0 : h)) * b_T + (b_T == 1 ? 0 : i)) * b_J + (b_J == 1 ? 0 : j);
    dt go = out_grad[out_idx];

    switch (op) {
    case '+':
        atomicAdd(&a_grad[a_idx], go);
        atomicAdd(&b_grad[b_idx], go);
        break;
    case '-':
        atomicAdd(&a_grad[a_idx], go);
        atomicAdd(&b_grad[b_idx], -go);
        break;
    case '*':
        atomicAdd(&a_grad[a_idx], go * b_data[b_idx]);
        atomicAdd(&b_grad[b_idx], go * a_data[a_idx]);
        break;
    case '/':
        atomicAdd(&a_grad[a_idx], go / b_data[b_idx]);
        atomicAdd(&b_grad[b_idx], go * (-a_data[a_idx] / (b_data[b_idx] * b_data[b_idx])));
        break;
    }
}





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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    // unravel i using precomputed input strides
    int idx[10]; // max rank = 10
    int flat_index = i;
    for (int d = 0; d < input_shape_size; ++d) {
        idx[d] = flat_index / d_input_strides[d];
        flat_index %= d_input_strides[d];
    }

    // Prepare output index
    int outidx[10];
    for (int d = 0; d < out_shape_size; ++d) {
        outidx[d] = idx[d];
    }
    if (keepdim) {
        outidx[input_shape_size - 1] = 0;  // last dim zeroed if keepdim
    }

    // ravel output index using precomputed strides
    int j = 0;
    for (int d = 0; d < out_shape_size; ++d) {
        j += outidx[d] * d_out_strides[d];
    }

    atomicAdd(&out_data[j], input_data[i] / reduce_size);
}


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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    int idx[10];
    int flat = i;

    // unravel input index
    for (int d = 0; d < input_shape_size; ++d) {
        idx[d] = flat / input_strides[d];
        flat %= input_strides[d];
    }

    // construct output index
    int out_idx[10];
    for (int d = 0; d < out_shape_size; ++d) {
        out_idx[d] = idx[d];
    }

    if (keepdim) {
        out_idx[input_shape_size - 1] = 0;  // last dim index 0
    }

    // ravel output index
    int j = 0;
    for (int d = 0; d < out_shape_size; ++d) {
        j += out_idx[d] * out_strides[d];
    }

    atomicAdd(&input_grad[i], out_grad[j] / reduce_size);
}




__global__ void square_kernel(
    const dt* __restrict__ input_data, 
    dt* __restrict__ out_data
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blockDim.x * gridDim.x) return;
    out_data[i] = input_data[i] * input_data[i];
}
__global__ void square_backward_kernel(
    const dt* __restrict__ input_data,
    dt* __restrict__ input_grad,
    const dt* __restrict__ out_grad
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gridDim.x * blockDim.x)return;
    atomicAdd(&input_grad[i], out_grad[i] * 2.0 * input_data[i]);
}

__global__ void sqrt_kernel(const dt* in, dt* out, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = sqrtf(in[i]);
    }
}
__global__ void sqrt_backward_kernel(const dt* in, const dt* grad_out, dt* grad_in, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dt val = in[i];
        dt val2 = (val > 1e-10f ? grad_out[i] * 0.5f / sqrtf(val) : 0.0f);
        if (isnan(val2) || isinf(val2)) {
            printf("Warning: NaN or Inf detected at sqrt_backward_kernel\n");
            val2 = 0; 
        }
        grad_in[i] += val2;
    }
}

__global__ void gather_kernel(
    const dt* __restrict__ weight,
    dt* __restrict__ out,
    const int* indices,
    int BT, int C) {

    int bt = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (bt >= BT || c >= C) return;

    int idx = indices[bt];
    out[bt * C + c] = __ldg(&weight[idx * C + c]);
}


__global__ void gather_backward_kernel(
    const dt* __restrict__ out_grad,
    const int* __restrict__ indices, 
    dt* __restrict__ weight_grad,
    int BT, int C)
{
    int bt = blockIdx.x;  
    int c = threadIdx.x;

    if (bt >= BT || c >= C) return;

    int idx = indices[bt];
    dt val = out_grad[bt * C + c];

    atomicAdd(&weight_grad[idx * C + c], val);
}


#if 0
__global__ void sum_kernel(const dt* __restrict__ input, dt* __restrict__ output, int size) {
    __shared__ dt sdata[512];  // adjust as per maxThreadsPerBlock

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    dt val = (i < size) ? input[i] : 0;
    sdata[tid] = val;
    __syncthreads();

    // block-wide reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void sum_backward_kernel(dt* __restrict__ grad_input, dt grad_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_input[i] += grad_output;
    }
}
#endif

__global__ void gelu_forward_kernel(const float* x, float* out, int N, bool approximate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float val = x[i];
    if (approximate) {
        const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
        float x3 = val * val * val;
        float tanh_arg = sqrt_2_over_pi * (val + 0.044715f * x3);
        float tanh_val = tanhf(tanh_arg);
        out[i] = 0.5f * val * (1.0f + tanh_val);
    }
    else {
        const float sqrt1_2 = 1.0f / sqrtf(2.0f);
        out[i] = 0.5f * val * (1.0f + erff(val * sqrt1_2));
    }
}

__global__ void gelu_backward_kernel(const float* x, const float* out_grad, float* x_grad, int N, bool approximate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float val = x[i];
    float grad = 0.0f;

    if (approximate) {
        const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
        float x2 = val * val;
        float x3 = x2 * val;
        float tanh_arg = sqrt_2_over_pi * (val + 0.044715f * x3);
        float tanh_val = tanhf(tanh_arg);
        float sech2 = 1.0f - tanh_val * tanh_val;

        float coeff = 0.5f * val * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x2);
        float dgelu_dx = 0.5f * (1.0f + tanh_val) + coeff;
        grad = dgelu_dx;
    }
    else {
        const float sqrt1_2 = 1.0f / sqrtf(2.0f);
        const float sqrt2_pi = sqrtf(2.0f / M_PI);
        float erf_term = erff(val * sqrt1_2);
        float exp_term = expf(-0.5f * val * val);
        float dgelu_dx = 0.5f * (1.0f + erf_term) + 0.5f * val * sqrt2_pi * exp_term;
        grad = dgelu_dx;
    }

    dt val2 = grad * out_grad[i];
    if (isnan(val2) || isinf(val2)) {
        printf("Warning: NaN or Inf detected at gelu_backward_kernel\n");
        val2 = 0;
    }
    x_grad[i] += val2;
}


// Attention
__global__ void scaled_dot_qk_softmax_kernel(
    const dt* __restrict__ qkv,
    dt* __restrict__ scores,
    dt* __restrict__ exps,
    dt* __restrict__ smax,
    int B, int H, int T, int C, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        bh = blockIdx.y,
        b = bh / H, h = bh % H;

    if (b >= B || h >= H || i >= T) return;

    int q_offset = head_dim * h,
        k_offset = C + head_dim * h,
        v_offset = 2 * C + head_dim * h,
        q_base = (T * b + i) * C,
        scores_base = ((b * H + h) * T + i) * T;

    dt row_max = -INFINITY;
    for (int j = 0; j <= i; j++) {
        int k_base = (T * b + j) * C;
        dt sum = 0.0;

        // Reinterpret the q and k slices (starting at respective offsets) as arrays of float4.
        // This assumes qkv is a float* pointing to interleaved query, key, value tensors.
        const float4* qv = reinterpret_cast<const float4*>(&qkv[q_base + q_offset]);
        const float4* kv = reinterpret_cast<const float4*>(&qkv[k_base + k_offset]);

        // Unroll the loop for better performance. Each iteration processes 4 float values at once.
        // Assumes head_dim is divisible by 4.
#pragma unroll
        for (int d = 0; d < head_dim / 4; d++) {
            float4 q = qv[d];
            float4 k = kv[d];

            sum += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
        }

        dt scaled = sum / sqrtf((dt)head_dim);
        scores[scores_base + j] = scaled;
        row_max = fmaxf(row_max, scaled);
    }

    dt sum_exp = 0.0;
    for (int j = 0; j <= i; j++) {
        int idx = scores_base + j;
        exps[idx] = expf(scores[idx] - row_max);
        sum_exp += exps[idx];
    }
    sum_exp = fmaxf(sum_exp, 1e-6f);

    for (int j = 0; j <= i; j++) {
        int idx = scores_base + j;
        smax[idx] = exps[idx] / sum_exp;
    }
}

__global__ void smax_weighted_v_kernel(
    const dt* __restrict__ qkv,
    const dt* __restrict__ smax,
    dt* __restrict__ v_out,
    int B, int H, int T, int C, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.y;
    int b = bh / H, h = bh % H;
    if (b >= B || h >= H || i >= T) return;

    int out_base = ((b * T) + i) * C + h * head_dim;
    int smax_base = ((b * H + h) * T + i) * T;

    for (int d = 0; d < head_dim; d++) {
        dt val = 0.0;
        for (int j = 0; j <= i; j++) {
            int s_idx = smax_base + j;
            int v_idx = (b * T + j) * (3 * C) + 2 * C + h * head_dim + d;
            val += smax[s_idx] * qkv[v_idx];
        }
        v_out[out_base + d] = val;
    }
}

__global__ void backward_attention_kernel(
    const dt* __restrict__ out_grad,
    const dt* __restrict__ smax,
    const dt* __restrict__ qkv,

    dt* __restrict__ smax_grad,
    dt* __restrict__ qkv_grad,
    dt* __restrict__ score_grad,

    int B, int n_head, int T, int C, int head_dim
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int bh = blockIdx.y;
    if (i >= T || bh >= B * n_head) return;

    int b = bh / n_head;
    int h = bh % n_head;

    int smax_base = ((b * n_head + h) * T + i) * T;
    int out_base = ((b * T) + i) * C + h * head_dim;
    int q_base = (b * T + i) * 3 * C + h * head_dim;

    // V
    for (int j = 0; j <= i; ++j) {
        int s_idx = smax_base + j;
        int v_idx = (b * T + j) * 3 * C + 2 * C + h * head_dim;

        for (int d = 0; d < head_dim; ++d) {
            dt grad_out = out_grad[out_base + d];
            dt v_val = qkv[v_idx + d];

            atomicAdd(&smax_grad[s_idx], grad_out * v_val);
            atomicAdd(&qkv_grad[v_idx + d], grad_out * smax[s_idx]);
        }
    }

    // Sum per row
    dt sum = 0.0f;
    for (int j = 0; j <= i; ++j) {
        int s_idx = smax_base + j;
        sum += smax_grad[s_idx] * smax[s_idx];
    }

    // Scores
    for (int j = 0; j <= i; ++j) {
        int s_idx = smax_base + j;
        int k_idx = (b * T + j) * 3 * C + C + h * head_dim;

        dt s = smax[s_idx];
        dt dL_ds = smax_grad[s_idx];
        dt d_softmax = s * (dL_ds - sum);
        if (isnan(d_softmax) || isinf(d_softmax)) {
            printf("Warning: NaN or Inf detected at backward_attention_kernel scores\n");
            d_softmax = 0;
        }
        atomicAdd(&score_grad[s_idx], d_softmax);

        // Q K
        for (int d = 0; d < head_dim; ++d) {
            dt grad_out = out_grad[out_base + d];
            dt s_val = s;
            dt q_val = qkv[q_base + d];
            dt k_val = qkv[k_idx + d];

            dt d_score = d_softmax / sqrtf((float)head_dim);
            atomicAdd(&qkv_grad[q_base + d], d_score * k_val);
            atomicAdd(&qkv_grad[k_idx + d], d_score * q_val);
        }
    }
}

__global__ void softmax_forward_kernel(
    const dt* __restrict__ input, 
    dt* __restrict__ out, 
    int B, int T, int V
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || t >= T) return;

    int base = (b * T + t) * V;

    // Step 1: compute max over V
    float max_val = input[base];
    for (int i = 0; i < V; i++) {
        max_val = fmaxf(max_val, input[base + i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        int idx = base + i;
        out[idx] = expf(input[idx] - max_val);
        sum += out[idx];
    }

    dt inv_sum = 1.0 / sum;
    for (int i = 0; i < V; i++) {
        out[base + i] *= inv_sum;
    }
}


__global__ void softmax_backward_kernel(
    const dt* __restrict__ out, 
    const dt* __restrict__ out_grad,
    dt* __restrict__ input_grad,
    int B, int T, int V
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || t >= T) return;

    int base = (b * T + t) * V;

    float dot = 0.0f;
    for (int i = 0; i < V; ++i)
        dot += out[base + i] * out_grad[base + i];

    for (int i = 0; i < V; ++i) {
        dt val = out[base + i] * (out_grad[base + i] - dot);
        if (isnan(val) || isinf(val)) {
            printf("Warning: NaN or Inf detected at softmax_backward_kernel\n");
            val = 0;
        }
        input_grad[base + i] = out[base + i] * (out_grad[base + i] - dot);
    }
}

__global__ void cross_entropy_kernel(
    const dt* __restrict__ input,
    dt* __restrict__ out,
    const int* __restrict__ targets,
    const int B, const int T, const int V
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || t >= T) return;

    int target_idx = targets[b * T + t];
    int idx = (b * T + t) * V + target_idx;
    dt prob = input[idx];

    atomicAdd(&out[0], -logf(prob + 1e-10));
}

__global__ void cross_entropy_backward_kernel(
    const dt* __restrict__ probs,      
    dt* __restrict__ grad_out,        
    const int* __restrict__ targets, 
    const int B, const int T, const int V,
    const dt grad_loss
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || t >= T) return;

    int target_idx = targets[b * T + t];
    int idx = (b * T + t) * V + target_idx;

    dt prob = probs[idx];
    dt val = (-1.0f / (prob + 1e-10f)) * (grad_loss / (B * T));

    // Check for NaN or Inf - print warning for debugging
    if (isnan(val) || isinf(val)) {
        printf("Warning: NaN or Inf detected at b=%d, t=%d, idx=%d, prob=%f, val=%f\n", b, t, idx, prob, val);
        val = 0; 
    }

    grad_out[idx] = val;
}


__global__ void sgd_step_kernel(dt* weights, dt* grads, float lr, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    dt updated = weights[i] - lr * grads[i];

    weights[i] = updated;
    grads[i] = 0.0f;
}



