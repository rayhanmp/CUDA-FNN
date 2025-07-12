#pragma once

__global__ void matvec_bias(float* W, float* x, float* b, float* y, int M, int K);
