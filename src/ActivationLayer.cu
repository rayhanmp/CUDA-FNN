#include "ActivationLayer.cuh"
#include <cuda_runtime.h>
#include <cmath>

__global__ void relu(float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = fmaxf(0.0f, y[idx]);
}

__global__ void sigmoid(float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 1.0f / (1.0f + expf(-y[idx]));
}

__global__ void tanh(float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = tanhf(y[idx]);
}

ActivationLayer::ActivationLayer(int in_dim, ActivationType type) : in_dim(in_dim), type(type) {}

ActivationLayer::~ActivationLayer() {}

void ActivationLayer::forward(float* input_d) {
    int threads = 256;
    int blocks = (in_dim + threads - 1) / threads;

    switch (type) {
        case ActivationType::ReLU:
            relu<<<blocks, threads>>>(input_d, in_dim);
            break;
        case ActivationType::Sigmoid:
            sigmoid<<<blocks, threads>>>(input_d, in_dim);
            break;
        case ActivationType::Tanh:
            tanh<<<blocks, threads>>>(input_d, in_dim);
            break;
    }

    cudaDeviceSynchronize();
}