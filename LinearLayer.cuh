#pragma once

class LinearLayer {
    public:
        LinearLayer(int in_dim, int out_dim);
        ~LinearLayer();

        void forward(const float* x_d, float* y_d);

    private:
        int in_dim;
        int out_dim;
        float* W_d;
        float* b_d;
};

__global__ void matvec_bias(float* W, float* x, float* b, float* y, int M, int K);

