#include "tensor.h"
#include <stdexcept>
#include <chrono>

bool global_cuda_enabled = false;


// CUDA management
void Tensor::alloc_gpu() {
    if (d_data || d_grad) free();

    cudaMalloc(&d_data, numel() * sizeof(dt));
    cudaMalloc(&d_grad, numel() * sizeof(dt));

    cudaMemset(d_data, 0, numel() * sizeof(dt));
    cudaMemset(d_grad, 0, numel() * sizeof(dt));
}

void Tensor::to_cpu() {
    if (d_data) {
        cudaMemcpy(data.data(), d_data, numel() * sizeof(dt), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        d_data = nullptr;
    }
    if (d_grad && requires_grad) {
        cudaMemcpy(grad.data(), d_grad, numel() * sizeof(dt), cudaMemcpyDeviceToHost);
        cudaFree(d_grad);
        d_grad = nullptr;
    }
}

void Tensor::to_gpu() {
    if (d_data || d_grad) free();

    cudaMalloc(&d_data, data.size() * sizeof(dt));
    cudaMemcpy(d_data, data.data(), numel() * sizeof(dt), cudaMemcpyHostToDevice);

    cudaMalloc(&d_grad, grad.size() * sizeof(dt));
    cudaMemcpy(d_grad, grad.data(), numel() * sizeof(dt), cudaMemcpyHostToDevice);
}

void Tensor::free() {
    if (d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    } 
    if (d_grad) {
        cudaFree(d_grad);
        d_grad = nullptr;
    }
}

void Tensor::free_all() {
    std::unordered_set<Tensor*> freed;
    for (Tensor* t : topo_order) {
        if (!t->constant_data && !freed.count(t)) {
            t->free();
            freed.insert(t);
        }
    }
}

void Tensor::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), 0.0);
    }
}

// Tensor pretty-printing with optional gradient
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    if (t.numel() == 1) {
        os << t.data[0];
        return os;
    }
    if (t.shape.empty()) {
        os << "[]";
        return os;
    }

    auto strides = compute_strides(t.shape);

    std::function<void(std::ostream&, size_t, size_t)> print_nd;
    print_nd = [&](std::ostream& os, size_t dim, size_t offset) {
        if (dim == t.shape.size()) {
            os << t.data[offset] << "(" << t.grad[offset] << ")";
            return;
        }

        os << "[";
        for (int i = 0; i < t.shape[dim]; ++i) {
            if (i > 0) {
                os << (dim < t.shape.size() - 1 ? ",\n " : ", ");
                for (size_t d = 0; d < dim; ++d) os << " ";
            }
            print_nd(os, dim + 1, offset + i * strides[dim]);
        }
        os << "]";
        };

    print_nd(os, 0, 0);
    return os;
}

// Backpropagation through computation graph
void Tensor::backward() {
    std::unordered_set<Tensor*> visited;

    // Topological sort
    std::function<void(Tensor*)> build_topo = [&](Tensor* t) {
        if (visited.count(t) || !t->requires_grad) return;
       
        visited.insert(t);
        for (Tensor* parent : t->prev) build_topo(parent);
        topo_order.push_back(t);
        };

    build_topo(this);
    std::reverse(topo_order.begin(), topo_order.end());

    // Seed gradient for root tensor
    if (numel() == 1)
        grad = std::vector<dt>(numel(), 1.0);

    // Execute backward functions
    for (Tensor* t : topo_order) {
   
        for (auto& backwardFunc : t->backwardFuncs) {
            if (backwardFunc) backwardFunc();
        }
        //print_gpu_data(t->d_grad, t->grad.size(), t->name);

        /*
        // Optional gradient clipping (CPU)
        if (t->requires_grad) {
            dt norm = 0;
            for (dt g : t->grad) norm += g * g;
            norm = std::sqrt(norm);

            if (norm > 4.0) {
                dt scale = 4.0 / norm;
                for (dt& g : t->grad) g *= scale;
            }
        }
        */
    }
}

