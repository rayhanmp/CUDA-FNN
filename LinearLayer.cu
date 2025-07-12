// linear-layer.cu

// Compute y = W * x

#include <iostream>
#include <cuda_runtime.h>
#include "LinearLayer.cuh"

LinearLayer::LinearLayer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim) {
    cudaMalloc(&W_d, in_dim * out_dim * sizeof(float));
    cudaMalloc(&b_d, out_dim * sizeof(float));

    // Initialize the weight and bias matrices with dummy values
    float* W_h = new float[in_dim * out_dim];
    float* b_h = new float[out_dim];

    std::srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < in_dim * out_dim; ++i) {
        W_h[i] = (std::rand() % 100) / 100.0f;
    }   

    for (int i = 0; i < out_dim; ++i) {
        b_h[i] = 0.0f;
    }

    cudaMemcpy(W_d, W_h, in_dim * out_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, out_dim * sizeof(float), cudaMemcpyHostToDevice);

    delete[] W_h;
    delete[] b_h;
}

LinearLayer::~LinearLayer() {
    cudaFree(W_d);
    cudaFree(b_d);
}

void LinearLayer::forward(const float* x_d, float* y_d) {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;
    matvec_bias<<<blocks, threads>>>(W_d, x_d, b_d, y_d, out_dim, in_dim);
    cudaDeviceSynchronize();
}

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

