    #include <iostream>
    #include <cuda_runtime.h>
    #include "LinearLayer.cuh"
    #include "ActivationLayer.cuh"

    int main() {
        const int in_dim = 4;
        const int hidden_dim = 5;
        const int out_dim = 3;

        float x_h[in_dim] = {1, 2, 3, 4};
        float y_h[out_dim];

        float *x_d, *h_d, *y_d;
        cudaMalloc(&x_d, in_dim     * sizeof(float));   // input
        cudaMalloc(&h_d, hidden_dim * sizeof(float));   // hidden
        cudaMalloc(&y_d, out_dim    * sizeof(float));   // output
        cudaMemcpy(x_d, x_h, in_dim * sizeof(float), cudaMemcpyHostToDevice);

        // Forward pass
        LinearLayer layer(in_dim, hidden_dim);
        LinearLayer layer2(hidden_dim, out_dim);
        ActivationLayer relu(hidden_dim, ActivationType::ReLU);

        layer.forward(x_d, h_d);
        relu.forward(h_d);
        layer2.forward(h_d, y_d);

        // Copy the output back to the host
        cudaMemcpy(y_h, y_d, out_dim * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Output:\n";
        for (int i = 0; i < out_dim; ++i) std::cout << "y[" << i << "] = " << y_h[i] << '\n';

        cudaFree(x_d);
        cudaFree(h_d);
        cudaFree(y_d);

        return 0;
    }