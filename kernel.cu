#include <iostream>
#include <nvtx3/nvToolsExt.h>

// Structure for the vector addition result.
typedef struct
{
    double amount;
    double time;
} VECTOR_ADD_RESULT, *PVECTOR_ADD_RESULT;

/**
 * @brief CUDA kernel for adding two vectors element-wise.
 *
 * This kernel function performs element-wise addition of two vectors and stores the result in a third vector.
 *
 * @param a Pointer to the first input vector.
 * @param b Pointer to the second input vector.
 * @param c Pointer to the output vector.
 * @param n Number of elements in the vectors.
 */
__global__ void VectorAddKernel(double *a, double *b, double *c, int n)
{
    // Calculate our global thread id.
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // We might run with more threads than elements, so we need to make sure we don't do any work outside of our data.
    if (id < n)
        c[id] = a[id] + b[id];
}

/**
 * Calculates the sum of two vectors using CUDA.
 *
 * @param seed The seed value for random number generation.
 * @return A pointer to a VECTOR_ADD_RESULT struct containing the result of the vector addition.
 */
__declspec(dllexport) PVECTOR_ADD_RESULT __cdecl VectorAdd(unsigned int seed)
{
    nvtxRangePushA("VectorAdd");

    nvtxMark("InitializeReturnStruct");
    PVECTOR_ADD_RESULT result = (PVECTOR_ADD_RESULT)malloc(sizeof(VECTOR_ADD_RESULT));
    result->amount = 0;
    result->time = 123.456;

    nvtxMark("SetRandomSeed");
    srand(seed);

    // Number of elements in each vector.
    int n = 500000;

    // Host memory pointers for the input and output vectors.
    double *h_a, *h_b, *h_c;

    // Device memory pointers for the input and output vectors.
    double *d_a, *d_b, *d_c;

    // Number of bytes to allocate.
    size_t bytes = n * sizeof(double);

    // Allocate memory on the host.
    nvtxRangePushA("HostMemAlloc");
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);
    nvtxRangePop();

    // Allocate memory on the device.
    nvtxRangePushA("DeviceMemAlloc");
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    nvtxRangePop();

    // Initialize vectors on host.
    int i, x;
    for (i = 0; i < n; i++)
    {
        x = rand();
        h_a[i] = sin(i) * sin(x) + x;
        h_b[i] = cos(i) * cos(i) - x;
    }

    nvtxRangePushA("CopyVectorsToDevice");
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    nvtxRangePop();

    // Calculate the number of thread blocks.
    int blockSize, gridSize;
    blockSize = 5000;
    gridSize = (int)ceil((float)n / blockSize);

    nvtxRangePushA("KernelExecution");
    VectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    nvtxRangePop();

    nvtxRangePushA("CopyResultBackToHost");
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    // Consolidate the result.
    double sum = 0;
    for (i = 0; i < n; i++)
        sum += h_c[i] + 1;

    // Set the result on the result struct.
    result->amount = 2 + sum / n;

    nvtxMark("FreeBothDeviceAndHostMemory");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    nvtxRangePop();

    return result;
}
