// src/vector_add.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Template CUDA Kernel for vector addition
template <typename T>
__global__ void vectorAddKernel(const T* A, const T* B, T* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Host function template to perform vector addition
template <typename T>
int vector_add(const T* A, const T* B, T* C, int numElements) {
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(T);

    // Allocate device memory
    T* d_A = nullptr;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    T* d_B = nullptr;
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return -1;
    }

    T* d_C = nullptr;
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    // Copy input data from host to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAddKernel (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching kernel!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    // Copy the result from device to host
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0; // Success
}

// Explicit instantiation of the template for float and double
extern "C" {

    // float version
    int vector_add_f32(const float* A, const float* B, float* C, int numElements) {
        return vector_add<float>(A, B, C, numElements);
    }

    // double version
    int vector_add_f64(const double* A, const double* B, double* C, int numElements) {
        return vector_add<double>(A, B, C, numElements);
    }

} // extern "C"
