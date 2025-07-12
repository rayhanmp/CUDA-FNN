// element-wise-squaring.cu

// Compute y[i] = x[i] * x[i] with each thread handles one index i

#include <iostream>
#include <cuda_runtime.h>

// Kernel function which runs on the GPU
__global__ void square(float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = x[i] * x[i];
    }
}

int main() {
    const int N = 100000;
    float x_h[N], y_h[N];
    std::cout << "Starting CUDA test...\n";

    // Initialize the input array
    for (int i = 0; i < N; i++) x_h[i] = i;

    // Allocate memory on the GPU
    float* x_d, *y_d;
    cudaMalloc(&x_d, N * sizeof(float));
    cudaMalloc(&y_d, N * sizeof(float));

    // Copy the input array to the GPU
    cudaMemcpy(x_d, x_h, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    square<<<blocks, threads>>>(x_d, y_d, N);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(y_h, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; ++i) std::cout << x_h[i] << "^2 = " << y_h[i] << '\n';

    // Free the memory
    cudaFree(x_d);
    cudaFree(y_d);

    return 0;
}