#pragma once

// SGD parameter update kernel
__global__ void sgd_update(float* param, const float* grad, float lr, int N);

// Matrix transpose vector multiplication kernel
__global__ void matT_vec(const float* W, const float* dy, float* dx, int out_dim, int in_dim);

// Outer product kernel
__global__ void outer_product(const float* dy, const float* x, float* dW, int out_dim, int in_dim); 

// Matrix-vector multiplication with bias kernel
__global__ void matvec_bias(const float* W, const float* x, const float* b, float* y, int M, int K);

// Compute MSE loss
void compute_mse_loss(const float* y_pred, const float* y_true, float* loss_d, int n);

// Compute MSE gradient
void compute_mse_grad(const float* y_pred, const float* y_true, float* grad_d, int n);