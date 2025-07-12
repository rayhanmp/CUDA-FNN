#include "relu.cuh"

// ReLU activation function

__global__ void relu(float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = fmaxf(0.0f, y[i]);
    }
}
