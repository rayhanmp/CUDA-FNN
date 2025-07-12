// test.cu
#include <iostream>
__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
