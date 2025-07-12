__global__ void sgd_update(float* param, const float* grad, float lr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) param[idx] -= lr * grad[idx];
}
