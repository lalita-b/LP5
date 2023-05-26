#include <stdio.h>
#include <cuda_runtime.h>

#define N 100000

__global__ void vectorAdd(int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    int size = N * sizeof(int);

    // Initialize input vectors a and b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // Copy input vectors from host to device memory
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Launch the vectorAdd kernel on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    // Copy the result vector from device to host memory
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            printf("Error: c[%d] != a[%d] + b[%d]\n", i, i, i);
            break;
        }
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
