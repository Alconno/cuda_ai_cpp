#include "tensor.h"
#include "kernels.h"

void Tensor::sqrt(Tensor& a) {
    this->resize_like(a);

    if (global_cuda_enabled) {
        // GPU sqrt forward
        this->alloc_gpu();

        dt* d_a_data = a.d_data;
        dt* d_out_data = this->d_data;

        int threads = 512;
        int blocks = (a.numel() + threads - 1) / threads;

        sqrt_kernel << <blocks, threads >> > (
            d_a_data,
            d_out_data,
            this->numel()
            );

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        // Optional: copy result to host
        // cudaMemcpy(this->data.data(), d_out_data, this->data.size() * sizeof(dt), cudaMemcpyDeviceToHost);

        /*
        std::vector<dt> cpu_ref(this->data.size());
        for (int i = 0; i < a.numel(); ++i) {
            cpu_ref[i] = std::sqrt(a.data[i]);
        }

        // Compare GPU vs CPU results
        bool match = true;
        for (int i = 0; i < a.numel(); ++i) {
            if (std::abs(cpu_ref[i] - this->data[i]) > 1e-5f) {
                std::cerr << "Mismatch at " << i << ": GPU=" << this->data[i] << " vs CPU=" << cpu_ref[i] << std::endl;
                match = false;
            }
        }
        if (match) std::cout << "[√] Forward GPU vs CPU matched.\n";
        */
    }
    else {
        // CPU sqrt
        for (int i = 0; i < a.numel(); ++i) {
            this->data[i] = std::sqrt(a.data[i]);
        }
    }

    // Backward logic
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

                sqrt_backward_kernel << <blocks, threads >> > (
                    d_a_data,
                    d_out_grad,
                    d_a_grad,
                    this->numel()
                    );

                cudaDeviceSynchronize();
                CHECK_CUDA(cudaGetLastError());

                // Optional: copy grad to host
                // cudaMemcpy(a.grad.data(), d_a_grad, a.grad.size() * sizeof(dt), cudaMemcpyDeviceToHost);

                /*
                std::vector<dt> cpu_ref_grad(a.numel());
                for (int i = 0; i < a.numel(); ++i) {
                    dt val = a.data[i];
                    cpu_ref_grad[i] = (val > 1e-10f ? this->grad[i] * 0.5f / std::sqrt(val) : 0.0f);
                }

                bool match = true;
                for (int i = 0; i < a.numel(); ++i) {
                    if (std::abs(cpu_ref_grad[i] - a.grad[i]) > 1e-5f) {
                        std::cerr << "Backward mismatch at " << i << ": GPU=" << a.grad[i] << " vs CPU=" << cpu_ref_grad[i] << std::endl;
                        match = false;
                    }
                }
                if (match) std::cout << "[√] Backward GPU vs CPU matched.\n";
                */
                });
        }
        else {
            this->backwardFuncs.push_back([this, &a]() {
                for (int i = 0; i < a.numel(); ++i) {
                    dt val = a.data[i];
                    if (val > 1e-10f) {
                        a.grad[i] += this->grad[i] * (0.5f / std::sqrt(val));
                    }
                    else {
                        a.grad[i] += 0.0f;
                    }
                }
                });
        }
    }
}