// src/vector_dot.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define the number of threads per block
#define THREADS_PER_BLOCK 256

// Template CUDA Kernel for partial dot product (reduction)
template <typename T>
__global__ void vectorDotKernel(const T* A, const T* B, T* partialSums, int numElements) {
    __shared__ T sharedData[THREADS_PER_BLOCK];
    
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes its partial sum
    T temp = 0;
    if (index < numElements) {
        temp = A[index] * B[index];
    }
    sharedData[tid] = temp;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < blockDim.x) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result for this block to partialSums
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

// Host function template to perform dot product
template <typename T>
int vector_dot(const T* A, const T* B, T* result, int numElements) {
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(T);

    // Allocate device memory for A and B
    T* d_A = nullptr;
    T* d_B = nullptr;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return -1;
    }

    // Copy input data from host to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    // Determine the number of blocks
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory for partial sums
    T* d_partialSums = nullptr;
    err = cudaMalloc((void**)&d_partialSums, blocksPerGrid * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device partial sums (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    // Launch the CUDA kernel
    vectorDotKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_partialSums, numElements);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorDotKernel (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_partialSums);
        return -1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching kernel!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_partialSums);
        return -1;
    }

    // Copy partial sums from device to host
    T* h_partialSums = (T*)malloc(blocksPerGrid * sizeof(T));
    if (h_partialSums == nullptr) {
        fprintf(stderr, "Failed to allocate host memory for partial sums!\n");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_partialSums);
        return -1;
    }

    err = cudaMemcpy(h_partialSums, d_partialSums, blocksPerGrid * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy partial sums from device to host (error code %s)!\n", cudaGetErrorString(err));
        free(h_partialSums);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_partialSums);
        return -1;
    }

    // Perform final reduction on host
    T dotProduct = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        dotProduct += h_partialSums[i];
    }
    *result = dotProduct;

    // Free host and device memory
    free(h_partialSums);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partialSums);

    return 0; // Success
}

// Explicit instantiation of the template for float and double
extern "C" {

    // float version
    int vector_dot_f32(const float* A, const float* B, float* result, int numElements) {
        return vector_dot<float>(A, B, result, numElements);
    }

    // double version
    int vector_dot_f64(const double* A, const double* B, double* result, int numElements) {
        return vector_dot<double>(A, B, result, numElements);
    }

} // extern "C"
