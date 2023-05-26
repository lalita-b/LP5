#include <stdio.h>

// Kernel function for matrix multiplication
__global__ void matrixMultiplication(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that the thread is within matrix bounds
    if (row < n && col < n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int n = 3;  // Size of matrix
    int a[n][n], b[n][n], c[n][n];
    int *dev_a, *dev_b, *dev_c;

    // Initialize matrices a and b
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = i * n + j;
            b[i][j] = i + j * n;
        }
    }

    // Allocate memory on device for matrices a, b, and c
    cudaMalloc((void **)&dev_a, n * n * sizeof(int));
    cudaMalloc((void **)&dev_b, n * n * sizeof(int));
    cudaMalloc((void **)&dev_c, n * n * sizeof(int));

    // Copy matrices a and b from host to device
    cudaMemcpy(dev_a, a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block_size(32, 32);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

    // Launch kernel function for matrix multiplication
    matrixMultiplication<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, n);

    // Copy matrix c from device to host
    cudaMemcpy(c, dev_c, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print resulting matrix c
    printf("Matrix C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // Free memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
