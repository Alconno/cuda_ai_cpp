#include "tensor.h"
#include "kernels.h"

// Broadcast two shapes into a common compatible shape
std::vector<int> broadcast_shapes(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    std::vector<int> s1 = shape1;
    std::vector<int> s2 = shape2;
    std::vector<int> result;

    while (s1.size() < s2.size()) s1.insert(s1.begin(), 1);
    while (s2.size() < s1.size()) s2.insert(s2.begin(), 1);

    for (size_t i = 0; i < s1.size(); ++i) {
        if (s1[i] == s2[i]) result.push_back(s1[i]);
        else if (s1[i] == 1) result.push_back(s2[i]);
        else if (s2[i] == 1) result.push_back(s1[i]);
        else throw std::runtime_error("Incompatible broadcast shapes");
    }

    return result;
}

void Tensor::resize_like(Tensor& other) {
    this->shape = broadcast_shapes(this->shape, other.shape);
    this->data.resize(product(this->shape));
    if (this->requires_grad || other.requires_grad)
        this->grad.resize(product(this->shape), 0.0);
}

// Convert flat index to multidimensional coordinates
std::vector<int> unravel_index(int flat_index, const std::vector<int>& shape) {
    std::vector<int> indices;
    indices.reserve(shape.size());

    int remaining = flat_index;
    for (size_t i = 0; i < shape.size(); ++i) {
        int stride = Tensor::product({ shape.begin() + i + 1, shape.end() });
        indices.push_back(remaining / stride);
        remaining %= stride;
    }

    return indices;
}

// Convert multidimensional coordinates to flat index
int ravel_index(const std::vector<int>& indices, const std::vector<int>& shape) {
    int flat_index = 0, stride = 1;
    int ndim = shape.size();
    int offset = ndim - indices.size();

    for (int i = ndim - 1; i >= 0; --i) {
        int idx = (i - offset >= 0) ? indices[i - offset] : 0;
        flat_index += idx * stride;
        stride *= shape[i];
    }

    return flat_index;
}

// Convert broadcasted indices back to original tensor's flat index
int broadcast_index(const std::vector<int>& indices,
    const std::vector<int>& orig_shape,
    const std::vector<int>& result_shape) {
    std::vector<int> aligned_indices;
    size_t offset = result_shape.size() - orig_shape.size();

    for (size_t i = 0; i < orig_shape.size(); ++i) {
        aligned_indices.push_back(orig_shape[i] == 1 ? 0 : indices[offset + i]);
    }

    return ravel_index(aligned_indices, orig_shape);
}

// Compute strides from a shape
std::vector<int> compute_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}


// Random
void print_gpu_data(const float* d_ptr, size_t size, const std::string& label) {
    std::vector<float> h_data(size);
    cudaMemcpy(h_data.data(), d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    if (!label.empty()) std::cout << label << ": ";
    std::cout << "[ ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "]\n";
}