#include <stdio.h>

#include "tensor.h"
#include <random>
#include <assert.h>
#include "Model.h"
#include <omp.h>

// Entry point for GPT training
int main() {
    // Set number of OpenMP threads (adjust to match your CPU)
    omp_set_num_threads(20);

    // Hyperparameters
    const int B = 32;         // Batch size
    const int T = 128;         // Sequence length
    const int C = 256;        // Embedding dimension
    const int vocab = 1024;   // Vocabulary size
    const int n_head = 8;     // Number of attention heads
    const int n_layer = 4;    // Number of transformer blocks

    // Model configuration
    ModelConfig config(
        vocab, T, C, n_layer, n_head,
        /*bias=*/false,
        /*dropout=*/0.0f,
        /*emb_std=*/0.02f,
        /*attn_std=*/1.0f / std::sqrt(C),
        /*mlp_std=*/std::sqrt(2.0f / C),
        /*ln_gamma_val=*/1.0f,
        /*linear_std=*/1.0f / std::sqrt(C)
    );

    // Model and optimizer setup
    Model model(config, /*device=*/1);  // 1 = GPU, 0 = CPU
    const int max_steps = 10'000;
    const int warmup_steps = max_steps * 0.1;
    const dt max_lr = 0.06;
    const dt min_lr = 0.01;
    Optimizer opt(max_steps, warmup_steps, max_lr, min_lr);

    // Reset CUDA before training
    cudaDeviceReset();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Start training
    model.train(max_steps, opt, B);

    // Reset CUDA after training
    cudaDeviceReset();

    return 0;
}