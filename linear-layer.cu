// linear-layer.cu

// Compute y = W * x

#include <iostream>
#include <cuda_runtime.h>

__global__ void matvec_bias(float* W, float* x, float* b, float* y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            sum += W[row * K + j] * x[j]; // Dot product of W[row]
        }
        y[row] = sum + b[row];
    }
}

int main() {
    const int M = 3; // Number of rows in W
    const int K = 4; // Number of columns in W

    // Initialize the weight matrix
    float W_h[M * K] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    // Initialize the input vector
    float x_h[K] = {1, 2, 3, 4};

    // Initialize the bias vector
    float b_h[M] = {0.5f, 1.0f, 1.5f};  // Example values
    
    // Allocate memory on the CPU
    float y_h[M];

    // Allocate memory on the GPU
    float *W_d, *x_d, *y_d, *b_d;
    cudaMalloc(&b_d, M * sizeof(float));
    cudaMalloc(&W_d, M * K * sizeof(float));
    cudaMalloc(&x_d, K * sizeof(float));
    cudaMalloc(&y_d, M * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(W_d, W_h, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_h, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, M * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    matvec_bias<<<blocks, threads>>>(W_d, x_d, b_d, y_d, M, K);
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(y_h, y_d, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < M; ++i) std::cout << "y[" << i << "] = " << y_h[i] << '\n';

    // Free the memory
    cudaFree(W_d);
    cudaFree(x_d);
    cudaFree(b_d);
    cudaFree(y_d);

    return 0;
}