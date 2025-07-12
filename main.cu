#include <iostream>
#include <cuda_runtime.h>
#include "LinearLayer.cuh"
#include "relu.cuh"

int main() {
    const int in_dim = 4;
    const int out_dim = 3;

    float x_h[in_dim] = {1, 2, 3, 4};
    float y_h[out_dim];

    float* x_d, *y_d;
    cudaMalloc(&x_d, in_dim * sizeof(float));
    cudaMalloc(&y_d, out_dim * sizeof(float));
    cudaMemcpy(x_d, x_h, in_dim * sizeof(float), cudaMemcpyHostToDevice);

    LinearLayer layer(in_dim, out_dim);
    layer.forward(x_d, y_d);

    cudaMemcpy(y_h, y_d, out_dim * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output:\n";
    for (int i = 0; i < out_dim; ++i)
        std::cout << "y[" << i << "] = " << y_h[i] << '\n';

    cudaFree(x_d);
    cudaFree(y_d);
    return 0;
}