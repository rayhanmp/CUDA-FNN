__global__ void outer_product(const float* dy, const float* x, float* dW, int out_dim, int in_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < out_dim && col < in_dim) {
        dW[row * in_dim + col] = dy[row] * x[col];
    }
}
