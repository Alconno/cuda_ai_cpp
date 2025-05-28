#include "tensor.h"

// Matrix ops ( Currently unused )
void Tensor::slice(int dim, int start, int end, std::shared_ptr<Tensor>& q) {
    // Adjust shape
    q->shape = this->shape;
    q->shape[dim] = end - start;
    int dims = q->shape.size();

    q->data_view.resize(q->numel());
    q->grad.resize(q->numel());

    int B = dims == 4 ? q->shape[0] : 1,
        H = dims == 3 ? q->shape[dims - 3] : 1,
        T = q->shape[dims - 2], J = q->shape[dims - 1];
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < J; j++) {
                    q->data_view[ravel_index({ b,h,i,j }, q->shape)] = &data[ravel_index({ b,h,i,j + start }, shape)];
                }
            }

}
Tensor Tensor::view(const std::vector<int>& new_shape) const {
    // 1. Calculate total elements
    int total = 1;
    for (int dim : new_shape) total *= dim;

    // 2. Handle -1 dimension
    std::vector<int> final_shape = new_shape;
    auto it = std::find(final_shape.begin(), final_shape.end(), -1);
    if (it != final_shape.end()) {
        int known = 1;
        for (int dim : final_shape)
            if (dim != -1) known *= dim;
        *it = numel() / known;
        total = numel();
    }

    // 3. Validate shape compatibility
    if (total != numel()) {
        throw std::runtime_error("Shape mismatch in view()");
    }

    // 4. Create new view
    Tensor result = *this;
    result.shape = final_shape;
    return result;
}
Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0) dim0 += shape.size();
    if (dim1 < 0) dim1 += shape.size();

    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim0], new_shape[dim1]);

    Tensor result(new_shape, {});
    result.requires_grad = this->requires_grad;

    auto old_strides = compute_strides(shape);       // for unraveling i
    auto new_strides = compute_strides(new_shape);   // for raveling new_index
    int total_dims = shape.size();

    for (int i = 0; i < numel(); i++) {
        std::vector<int> indices(shape.size());
        int remainder = i;

        // Unravel index
        for (int j = 0; j < shape.size(); j++) {
            indices[j] = remainder / old_strides[j];
            remainder %= old_strides[j];
        }

        std::swap(indices[dim0], indices[dim1]);

        // RAVEL using new_strides
        int new_index = 0;
        for (int j = 0; j < shape.size(); j++) {
            new_index += indices[j] * new_strides[j];
        }

        result.data[new_index] = data[i];
    }

    return result;
}