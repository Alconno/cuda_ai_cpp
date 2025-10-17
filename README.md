# Custom CUDA Transformer

This project is a from-scratch implementation of a transformer neural network in C++ with CUDA acceleration. It includes a custom autograd system and all necessary tensor operations.

## Features

- Embedding layers with token and positional embeddings  
- Multi-head self-attention with scaled dot-product attention  
- Layer normalization and MLP feed-forward layers  
- Autograd with backward functions implemented via lambdas  
- Support for CPU and GPU (CUDA) execution  
- Manual tensor indexing for performance and simplicity  
- Training using cross-entropy loss and softmax output  
- Benchmarking throughput and learning on synthetic token patterns  

## Build & Run

### Requirements

- CUDA Toolkit (tested with version v12.5)  
- Visual Studio with CUDA integration  
- C++17 compiler support  

### Build

1. Clone the repository  
2. Open the solution file in Visual Studio  
3. Ensure CUDA is properly configured in project properties  
4. Build the project  

### Run

- Press Run in Visual Studio to start training and benchmarking  
- Output will show tokens per second and training progress on synthetic data  

## Performance

- Current speed is ~2000 tokens/sec for 11 million parameters  
- Backward pass bottlenecked by atomic operations in CUDA kernels  
- Future improvements planned to remove atomics and enable mixed precision  

## Notes

- Model operates entirely on GPU after initialization to minimize CPU-GPU data transfers  
- Broadcasting support limited; manual flat indexing used  
- Uses float32 for tensor values  

## Future Work

- Remove/minimize atomic operations for faster backward pass  
- Implement mixed precision training   
- Improve CUDA kernel optimizations (warp shuffling, shared memory)  
