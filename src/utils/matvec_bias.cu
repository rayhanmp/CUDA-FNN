__global__ void matvec_bias(const float* W, const float* x, const float* b, float* y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            sum += W[row * K + j] * x[j]; // Dot product of W[row]
        }
        y[row] = sum + b[row];
    }
}
