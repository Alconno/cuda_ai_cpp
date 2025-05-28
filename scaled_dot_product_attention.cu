#include "tensor.h"
#include "kernels.h"



void Tensor::scaled_dot_product_attention(std::unordered_map<std::string, Tensor>& mats) {
    // this = qkv

    int B = mats["scores"].shape[0], n_head = mats["scores"].shape[1],
        T = mats["scores"].shape[2], C = mats["v_out"].shape[2], head_dim = C / n_head;

    assert(C % n_head == 0);

    auto AM = [&](char c) -> const int {return c == 'k' ? C * 1 : c == 'v' ? C * 2 : 0; }; // Index QKV mat

    Tensor& scores = mats["scores"],
        & exps = mats["exps"],
        & smax = mats["smax"],
        & v_out = mats["v_out"];


    if (global_cuda_enabled) {
        // GPU 
        scores.alloc_gpu();
        exps.alloc_gpu();
        smax.alloc_gpu();
        v_out.alloc_gpu();

        dt* d_qkv = this->d_data, * d_scores = scores.d_data, * d_exps = exps.d_data,
            * d_smax = smax.d_data, * d_v_out = v_out.d_data;

        dim3 block(8, 32);
        int grid_x = (T + block.x - 1) / block.x;
        int grid_y = B * n_head;
        dim3 grid(grid_x, grid_y, 1);

        scaled_dot_qk_softmax_kernel << <grid, block >> > (
            d_qkv, d_scores, d_exps, d_smax,
            B, n_head, T, C, head_dim
            );

        cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());

        smax_weighted_v_kernel << <grid, block >> > (
            d_qkv, d_smax, d_v_out,
            B, n_head, T, C, head_dim
            );

        cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());

        //cudaMemcpy(scores.data.data(), scores.d_data, scores.data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
        //cudaMemcpy(exps.data.data(), exps.d_data, exps.data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
        //cudaMemcpy(smax.data.data(), smax.d_data, smax.data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
        //cudaMemcpy(v_out.data.data(), v_out.d_data, v_out.data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
    }
    else {
        // CPU
        std::vector<int> scores_strides = compute_strides(scores.shape),
            exps_strides = compute_strides(exps.shape),
            smax_strides = compute_strides(smax.shape),
            vout_strides = compute_strides(v_out.shape);

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < n_head; h++) {
                int q_offset = head_dim * h,
                    k_offset = C + head_dim * h,
                    v_offset = 2 * C + head_dim * h;

                for (int i = 0; i < T; i++) {
                    int q_base = (T * b + i) * C;

                    // Q * K computation
                    std::vector<int>score_indices;
                    for (int j = 0; j <= i; j++) {
                        int k_base = (T * b + j) * C;
                        score_indices.push_back(b * scores_strides[0] + h * scores_strides[1] + i * scores_strides[2] + j);

                        for (int d = 0; d < head_dim; d++)
                            scores.data[score_indices.back()] += this->data[q_base + q_offset + d] * this->data[k_base + k_offset + d];

                        scores.data[score_indices.back()] /= std::sqrt((dt)head_dim);
                    }

                    // Softmax
                    dt row_max = -INFINITY;
                    for (int j = 0; j <= i; j++) {
                        row_max = std::max(row_max, scores.data[score_indices[j]]);
                    }

                    dt sum_exp = 0.0;
                    std::vector<int>exp_indices;
                    for (int j = 0; j <= i; j++) {
                        exp_indices.push_back(b * exps_strides[0] + h * exps_strides[1] + i * exps_strides[2] + j);
                        exps.data[exp_indices.back()] = std::exp(scores.data[score_indices[j]] - row_max);
                        sum_exp += exps.data[exp_indices.back()];
                    }
                    sum_exp = std::max(sum_exp, 1e-6f);

                    for (int j = 0; j <= i; j++) {
                        int smax_idx = b * smax_strides[0] + h * smax_strides[1] + i * smax_strides[2] + j;
                        smax.data[smax_idx] = exps.data[exp_indices[j]] / sum_exp;
                    }
                }



                // smax @ V
                for (int i = 0; i < T; i++) {
                    for (int j = 0; j < head_dim; j++) {
                        int out_idx = b * vout_strides[0] + i * vout_strides[1] + h * head_dim + j;
                        for (int k = 0; k <= i; k++) {
                            int smax_idx = b * smax_strides[0] + h * smax_strides[1] + i * smax_strides[2] + k,
                                v_idx = (T * b + k) * (3 * C) + AM('v') + head_dim * h + j;

                            v_out.data[out_idx] += smax.data[smax_idx] * this->data[v_idx];
                        }
                    }
                }
            }
        }
    }



    if (this->requires_grad) {
        v_out.prev.insert(this);
        v_out.prev.insert(&mats["smax"]);
        v_out.prev.insert(&mats["exps"]);
        v_out.prev.insert(&mats["scores"]);

        if (global_cuda_enabled) {
            v_out.backwardFuncs.push_back([&mats, this]() {
                Tensor& v_out = mats["v_out"],
                & smax = mats["smax"],
                & exps = mats["exps"],
                & scores = mats["scores"];

            /*
            std::vector<dt> cpu_qkv_data = this->data;
            std::vector<dt> cpu_qkv_grad = this->grad;
            std::vector<dt> cpu_smax_data = smax.data;
            std::vector<dt> cpu_smax_grad = smax.grad;
            std::vector<dt> cpu_vout_grad = v_out.grad;
            std::vector<dt> cpu_scores_grad = scores.grad;
            */

            int B = v_out.shape[0], T = v_out.shape[1], C = v_out.shape[2];
            int n_head = smax.shape[1], head_dim = C / n_head;

            dt* d_grad_vout = v_out.d_grad, * d_grad_smax = smax.d_grad, * d_grad_qkv = this->d_grad,
                * d_grad_scores = scores.d_grad;
            cudaMemset(d_grad_smax, 0, smax.grad.size() * sizeof(dt));
            cudaMemset(d_grad_qkv, 0, this->grad.size() * sizeof(dt));
            cudaMemset(d_grad_scores, 0, scores.grad.size() * sizeof(dt));

            dt* d_smax = smax.d_data, * d_qkv = this->d_data;

            dim3 block(512);
            int grid_x = (T + block.x - 1) / block.x;
            int grid_y = B * n_head;
            dim3 grid(grid_x, grid_y);

            backward_attention_kernel << <grid, block >> > (
                d_grad_vout,
                d_smax,
                d_qkv,
                d_grad_smax,
                d_grad_qkv,
                d_grad_scores,
                B, n_head, T, C, head_dim
                );
            cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());


            cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());

            //cudaMemcpy(smax.grad.data(), d_grad_smax, smax.grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);
            //cudaMemcpy(this->grad.data(), d_grad_qkv, this->grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);
            //cudaMemcpy(scores.grad.data(), d_grad_scores, scores.grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);

            /*
            auto AM = [&](char c) -> const int {
                return c == 'k' ? C * 1 : c == 'v' ? C * 2 : 0;
                };

            std::vector<int> smax_strides = compute_strides(smax.shape);
            std::vector<int> vout_strides = compute_strides(v_out.shape);
            std::vector<int> scores_strides = compute_strides(scores.shape);

            for (int b = 0; b < B; b++) {
                for (int h = 0; h < n_head; h++) {
                    for (int i = 0; i < T; i++) {
                        for (int j = 0; j < head_dim; j++) {
                            int out_idx = b * vout_strides[0] + i * vout_strides[1] + h * head_dim + j;
                            dt grad_out = cpu_vout_grad[out_idx];

                            for (int k = 0; k <= i; k++) {
                                int smax_idx = b * smax_strides[0] + h * smax_strides[1] + i * smax_strides[2] + k;
                                int v_idx = (T * b + k) * (3 * C) + AM('v') + h * head_dim + j;


                                cpu_smax_grad[smax_idx] += grad_out * cpu_qkv_data[v_idx];
                                cpu_qkv_grad[v_idx] += grad_out * cpu_smax_data[smax_idx];
                            }
                        }
                    }
                    for (int i = 0; i < T; i++) {
                        for (int j = 0; j <= i; j++) {
                            // Calculate total_sum for current i
                            dt total_sum = 0.0f;
                            for (int k = 0; k <= i; k++) {
                                int smax_idx = b * smax_strides[0] + h * smax_strides[1] + i * smax_strides[2] + k;
                                total_sum += cpu_smax_data[smax_idx] * cpu_smax_grad[smax_idx];
                            }

                            int score_idx = b * scores_strides[0] + h * scores_strides[1] + i * scores_strides[2] + j;
                            dt s_val = cpu_smax_data[score_idx];
                            dt dL_ds = cpu_smax_grad[score_idx];

                            dt grad_score = s_val * (dL_ds - total_sum);

                            cpu_scores_grad[score_idx] += grad_score;
                        }



                        for (int j = 0; j <= i; j++) {
                            int score_idx = b * scores_strides[0] + h * scores_strides[1] + i * scores_strides[2] + j;
                            dt d_score = cpu_scores_grad[score_idx] / std::sqrt(head_dim);

                            for (int k = 0; k < head_dim; k++) {
                                int q_idx = (T * b + i) * (3 * C) + AM('q') + h * head_dim + k;
                                int k_idx = (T * b + j) * (3 * C) + AM('k') + h * head_dim + k;

                                dt q_val = cpu_qkv_data[q_idx];
                                dt k_val = cpu_qkv_data[k_idx];

                                // Backprop into Q and K tensors
                                cpu_qkv_grad[q_idx] += d_score * k_val;
                                cpu_qkv_grad[k_idx] += d_score * q_val;
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < cpu_qkv_grad.size(); i++) {

                std::cout << "\nqkv: " << cpu_qkv_grad[i] << " != " << this->grad[i] << std::endl;
                std::cout << "smax: " << cpu_smax_grad[i] << " != " << smax.grad[i] << std::endl;
                std::cout << "scores: " << cpu_scores_grad[i] << " != " << scores.grad[i] << std::endl;

            }
             */
                });
        }
        else {
            v_out.backwardFuncs.push_back([&mats, this]() {
                Tensor& d_v_out = mats["v_out"],
                & smax = mats["smax"],
                & exps = mats["exps"],
                & scores = mats["scores"];

            int B = d_v_out.shape[0], T = d_v_out.shape[1], C = d_v_out.shape[2];
            int n_head = smax.shape[1], head_dim = C / n_head;

            auto AM = [&](char c) -> const int { return c == 'k' ? C * 1 : c == 'v' ? C * 2 : 0; };

            std::vector<int> scores_strides = compute_strides(scores.shape),
                exps_strides = compute_strides(exps.shape),
                smax_strides = compute_strides(smax.shape),
                vout_strides = compute_strides(d_v_out.shape);

            for (int b = 0; b < B; b++) {
                for (int h = 0; h < n_head; h++) {
                    for (int i = 0; i < T; i++) {
                        for (int j = 0; j < head_dim; j++) {
                            int out_idx = b * vout_strides[0] + i * vout_strides[1] + h * head_dim + j;
                            dt grad_out = d_v_out.grad[out_idx];

                            for (int k = 0; k <= i; k++) {
                                // Calculate the indices for softmax and V
                                int smax_idx = b * smax_strides[0] + h * smax_strides[1] + i * smax_strides[2] + k;
                                int v_idx = (T * b + k) * (3 * C) + AM('v') + h * head_dim + j;

                                // Backpropagate through softmax and V
                                smax.grad[smax_idx] += grad_out * this->data[v_idx];
                                this->grad[v_idx] += grad_out * smax.data[smax_idx];
                            }
                        }
                    }

                    // Backprop through softmax (gradient for softmax inputs)
                    for (int i = 0; i < T; i++) {
                        for (int j = 0; j <= i; j++) {
                            int score_idx = b * scores_strides[0] + h * scores_strides[1] + i * scores_strides[2] + j;
                            dt grad_score = 0.0;

                            for (int k = 0; k <= i; k++) {
                                int softmax_out_idx = b * smax_strides[0] + h * smax_strides[1] + i * smax_strides[2] + k;
                                dt sj = smax.data[softmax_out_idx];
                                dt dL_ds = smax.grad[softmax_out_idx];

                                if (j == k)
                                    grad_score += dL_ds * smax.data[score_idx] * (1 - smax.data[score_idx]);
                                else
                                    grad_score -= dL_ds * smax.data[score_idx] * sj;
                            }

                            scores.grad[score_idx] += grad_score;
                        }


                        // Backprop into Q and K (gradients for Q and K)
                        for (int j = 0; j <= i; j++) {
                            int score_idx = b * scores_strides[0] + h * scores_strides[1] + i * scores_strides[2] + j;
                            dt d_score = scores.grad[score_idx] / std::sqrt(head_dim);

                            for (int k = 0; k < head_dim; k++) {
                                int q_idx = (T * b + i) * (3 * C) + AM('q') + h * head_dim + k;
                                int k_idx = (T * b + j) * (3 * C) + AM('k') + h * head_dim + k;

                                dt q_val = this->data[q_idx];
                                dt k_val = this->data[k_idx];

                                // Backprop into Q and K tensors
                                this->grad[q_idx] += d_score * k_val;
                                this->grad[k_idx] += d_score * q_val;
                            }
                        }

                    }
                }
            }
                });
        }
    }

}