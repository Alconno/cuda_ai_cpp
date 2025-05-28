#include "tensor.h"
#include "kernels.h"


void Tensor::weighted_sum(Tensor& a, Tensor& b) {
    int max_dims = std::max(2, (int)std::max(a.shape.size(), b.shape.size()));

    while (a.shape.size() != max_dims) a.shape.insert(begin(a.shape), 1);
    while (b.shape.size() != max_dims) b.shape.insert(begin(b.shape), 1);

    // Compute the shape of the result tensor
    this->shape = a.shape;

    if (a.shape.back() == b.shape.back()) {
        int C = a.shape.back(); // shared dim
        int b_rows = b.data.size() / C;
        this->shape.back() = b.shape[b.shape.size() - 2];
        this->data.resize(this->numel());
        this->grad.resize(this->numel());

        // Tensor test_cpu = *this;

        int B = max_dims == 4 ? shape[0] : 1,
            H = max_dims == 3 ? shape[max_dims - 3] : 1,
            T = shape[max_dims - 2], J = shape[max_dims - 1];

        if (global_cuda_enabled) {
            // GPU dot product forward pass
            this->alloc_gpu();

            dt* d_a = a.d_data, * d_b = b.d_data, * d_out = this->d_data;
            size_t size_a = a.numel() * sizeof(dt);
            size_t size_b = b.numel() * sizeof(dt);
            size_t size_out = this->numel() * sizeof(dt);

            dim3 block(16, 16);
            int grid_x = (T + block.x - 1) / block.x;
            int grid_y = (J + block.y - 1) / block.y;
            int grid_z = B * H;
            dim3 grid(grid_x, grid_y, grid_z);

            weighted_sum_kernel << <grid, block >> > (d_a, d_b, d_out, B, H, T, J, C, b_rows);

            cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());

            // Copy result back to host
            //cudaMemcpy(data.data(), d_data, data.size() * sizeof(dt), cudaMemcpyDeviceToHost);
        }
        else {
            // CPU dot product forward pass
#pragma omp parallel for collapse(4) schedule(static)
            for (int batch = 0; batch < B; batch++) {
                int batch_offset_i = batch * H * T;
                int batch_offset_j = batch * H * J;
                int batch_offset_out = batch * H * T * J;

                for (int head = 0; head < H; head++) {
                    int head_offset_i = batch_offset_i + head * T;
                    int head_offset_j = batch_offset_j + head * J;
                    int head_offset_out = batch_offset_out + head * T * J;

                    for (int i = 0; i < T; i++) {
                        int i_idx = (head_offset_i + i) * C;
                        int out_idx = head_offset_out + i * J;

                        for (int j = 0; j < J; j++) {
                            int jj = (head_offset_j + j) % b_rows;
                            int j_idx = jj * C;

                            //printf("%d, %d, %d\n", i_idx, j_idx, out_idx); //  i=16 j=240 b=0 h=0 => result: 0.911536
                            dt sum = 0.0;
                            int k = 0;
                            for (; k + 3 < C; k += 4) {
                                sum += a.data[i_idx + k + 0] * b.data[j_idx + k + 0];
                                sum += a.data[i_idx + k + 1] * b.data[j_idx + k + 1];
                                sum += a.data[i_idx + k + 2] * b.data[j_idx + k + 2];
                                sum += a.data[i_idx + k + 3] * b.data[j_idx + k + 3];
                            }
                            for (; k < C; k++) {
                                sum += a.data[i_idx + k] * b.data[j_idx + k];
                            }


                            this->data[out_idx + j] = sum;

                            /*
                            test_cpu.data[out_idx + j] = sum;
                            float cpudata = test_cpu.data[out_idx + j];
                            float gpudata = data[out_idx + j];
                            if (std::abs(cpudata - gpudata) > 1e-5f) {
                                std::cout << cpudata << " != " << gpudata << std::endl;
                                std::cout << out_idx + j << " is wrong\n";
                            }
                            */
                        }
                    }
                }
            }
        }
    }


    // Backward closure
    this->prev.insert(&a);
    this->prev.insert(&b);


    this->backwardFuncs.push_back([&a, &b, this]() {

        int max_dims = std::max(a.shape.size(), b.shape.size());
        int B = max_dims == 4 ? shape[0] : 1,
            H = max_dims == 3 ? shape[max_dims - 3] : 1,
            T = shape[max_dims - 2],
            J = shape[max_dims - 1],
            C = a.shape.back(); // shared last dim

        if (global_cuda_enabled) {
            // GPU dot product backward pass
            dt* d_a = a.d_data, * d_b = b.d_data,
                * d_a_grad = a.d_grad, * d_b_grad = b.d_grad,
                * d_out_grad = this->d_grad;

            dim3 block(16, 16);
            int grid_x = (T + block.x - 1) / block.x;
            int grid_y = (J + block.y - 1) / block.y;
            int grid_z = B * H;
            dim3 grid(grid_x, grid_y, grid_z);


            weighted_sum_backward_kernel << <grid, block >> >
                (d_a, d_b, d_a_grad, d_b_grad, d_out_grad, B, H, T, J, C, b.data.size() / C);

            cudaDeviceSynchronize(); CHECK_CUDA(cudaGetLastError());

            //cudaMemcpy(a.grad.data(), d_a_grad, a.grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);
            //cudaMemcpy(b.grad.data(), d_b_grad, b.grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);
        }
        else {
            // CPU dot prod backward pass
#pragma omp parallel for collapse(4) schedule(static)
            for (int batch = 0; batch < B; batch++) {
                int batch_offset_i = batch * H * T;
                int batch_offset_j = batch * H * J;
                int batch_offset_out = batch * H * T * J;

                for (int head = 0; head < H; head++) {
                    int head_offset_i = batch_offset_i + head * T;
                    int head_offset_j = batch_offset_j + head * J;
                    int head_offset_out = batch_offset_out + head * T * J;

                    for (int i = 0; i < T; i++) {
                        int i_idx = (head_offset_i + i) * C;
                        int out_idx = head_offset_out + i * J;

                        for (int j = 0; j < J; j++) {
                            int jj = (head_offset_j + j) % (b.data.size() / C);
                            int j_idx = jj * C;

                            dt d_out = this->grad[out_idx + j];

                            int k = 0;
                            for (; k + 3 < C; k += 4) {
                                a.grad[i_idx + k + 0] += d_out * b.data[j_idx + k + 0];
                                a.grad[i_idx + k + 1] += d_out * b.data[j_idx + k + 1];
                                a.grad[i_idx + k + 2] += d_out * b.data[j_idx + k + 2];
                                a.grad[i_idx + k + 3] += d_out * b.data[j_idx + k + 3];

                                b.grad[j_idx + k + 0] += d_out * a.data[i_idx + k + 0];
                                b.grad[j_idx + k + 1] += d_out * a.data[i_idx + k + 1];
                                b.grad[j_idx + k + 2] += d_out * a.data[i_idx + k + 2];
                                b.grad[j_idx + k + 3] += d_out * a.data[i_idx + k + 3];
                            }
                            for (; k < C; k++) {
                                a.grad[i_idx + k] += d_out * b.data[j_idx + k];
                                b.grad[j_idx + k] += d_out * a.data[i_idx + k];
                            }
                        }
                    }
                }
            }
        }
        });
}