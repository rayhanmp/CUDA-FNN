__global__ void matT_vec(const float* W, const float* dy, float* dx, int out_dim, int in_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < in_dim) {
        float sum = 0.0f;
        for (int j = 0; j < out_dim; ++j) {
            sum += W[j * in_dim + idx] * dy[j];
        }
        dx[idx] = sum;
    }
}