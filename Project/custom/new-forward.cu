#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 19
#define BLOCK_SIZE 32
#define Height_out (Height - K + 1)
#define Width_out  (Width  - K + 1)
#define Height_grid ((int)ceil(1.0 * Height_out / TILE_WIDTH))
#define Width_grid  ((int)ceil(1.0 * Width_out  / TILE_WIDTH))
#define W_unroll (Height_out * Width_out)
#define H_unroll (Channel * K * K)
#define CONST_SIZE 6666

// Weight matrix (kernel values) in constant memory (all of the following)
__constant__ float const_mask[CONST_SIZE];

// baseline conv kernel
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / Width_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % Width_grid) * TILE_WIDTH + threadIdx.x;

    if (h < Height_out && w < Width_out) {
        float res = 0.0;
        for (int c = 0; c < Channel; c++)
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    res += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
        out_4d(b, m, h, w) = res;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

// Tiled shared memory convolution
__global__ void conv_forward_kernel_shared(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    extern __shared__ float input_tile[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile_3d(i2, i1, i0) input_tile[(i2) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)) + (i1) * (TILE_WIDTH + K - 1) + i0]
    // Insert your GPU convolution kernel code here
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / Width_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.y % Width_grid) * TILE_WIDTH + tx;

    int h_start = h - ty;
    int w_start = w - tx;

    // pre-load the input array into shared memory
    for (int c = 0; c < Channel; c++)
        for (int p = ty; p < TILE_WIDTH + K - 1; p += TILE_WIDTH)
            for (int q = tx; q < TILE_WIDTH + K - 1; q += TILE_WIDTH)
                tile_3d(c, p, q) = in_4d(b, c, h_start + p, w_start + q);

    __syncthreads();
    if (h < Height_out && w < Width_out) {
        float res = 0.0;
        for (int c = 0; c < Channel; c++)
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    res += tile_3d(c, ty + p, tx + q) * mask_4d(m, c, p, q);
        out_4d(b, m, h, w) = res;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
}

// Input channel reduction: atomics
__global__ void conv_forward_kernel_atomics(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    extern __shared__ float input_tile[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile_3d(i2, i1, i0) input_tile[(i2) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)) + (i1) * (TILE_WIDTH + K - 1) + i0]
    // Insert your GPU convolution kernel code here
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int c  = threadIdx.z;

    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / Width_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.y % Width_grid) * TILE_WIDTH + tx;

    int h_start = h - ty;
    int w_start = w - tx;

    // pre-load the input array into shared memory
    for (int p = ty; p < TILE_WIDTH + K - 1; p += TILE_WIDTH)
        for (int q = tx; q < TILE_WIDTH + K - 1; q += TILE_WIDTH)
            tile_3d(c, p, q) = in_4d(b, c, h_start + p, w_start + q);

    __syncthreads();
    if (h < Height_out && w < Width_out) {
        float res = 0.0;
        for (int p = 0; p < K; p++)
            for (int q = 0; q < K; q++)
                res += tile_3d(c, ty + p, tx + q) * mask_4d(m, c, p, q);
        // out_4d(b, m, h, w) = res;
        atomicAdd(&out_4d(b, m, h, w), res);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
}

// Input channel reduction: tree
__global__ void conv_forward_kernel_tree(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    extern __shared__ float sum[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tree_3d(i2, i1, i0) sum[(i2) * TILE_WIDTH * Channel + (i1) * Channel + i0]
    // Insert your GPU convolution kernel code here
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int c  = threadIdx.z;

    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / Width_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.y % Width_grid) * TILE_WIDTH + tx;

    if (h < Height_out && w < Width_out) {
        float res = 0.0;
        for (int p = 0; p < K; p++)
            for (int q = 0; q < K; q++)
                res += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
        tree_3d(ty, tx, c) = res;

        for (int stride = ceil(1.0*Channel/2); stride >= 1; stride >>= 1) {
            __syncthreads();
            if (c < stride && c + stride < Channel)  
                tree_3d(ty, tx, c) += tree_3d(ty, tx, c + stride);
        }
        __syncthreads();
        if (c == 0)  out_4d(b, m, h, w) = tree_3d(ty, tx, c);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tree_3d
}

// Shared memory matrix multiplication and input matrix unrolling
__global__ void unroll(const float *x, float *X_unroll, int b, const int Channel, const int Height, const int Width, const int K) {
    #define x4d(i3, i2, i1, i0) x[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < Channel * W_unroll) {
        int c = idx / W_unroll;
        int h_out = (idx % W_unroll) / Width_out;
        int w_out = (idx % W_unroll) % Width_out;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = c * K * K + p * K + q;
                int w_unroll = h_out * Width_out + w_out;
                X_unroll[h_unroll * W_unroll + w_unroll] = x4d(b, c, h_out + p, w_out + q);
            }
        }
    }
    #undef x4d
}

__global__ void matrixMultiply(float *X_unroll, float *y, int b, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedX[TILE_WIDTH][TILE_WIDTH];

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float result = 0;

    for (int i = 0; i < (H_unroll + TILE_WIDTH - 1) / TILE_WIDTH; i++) {

        if (i * TILE_WIDTH + tx < H_unroll && row < Map_out) {
            sharedW[ty][tx] = const_mask[row * H_unroll + i * TILE_WIDTH + tx];
        } else {
            sharedW[ty][tx] = 0.0;
        }

        if (i * TILE_WIDTH + ty < H_unroll && col < W_unroll) {
            sharedX[ty][tx] = X_unroll[(i * TILE_WIDTH + ty) * W_unroll + col];
        } else {
            sharedX[ty][tx] = 0.0;
        }
        
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++) {
            result += (sharedW[ty][j] * sharedX[j][tx]);
        }
        __syncthreads();

        if (row < Map_out && col < W_unroll) {
            y[b * Map_out * W_unroll + row * W_unroll + col] = result;
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int numOutput = Batch * Map_out * Width_out * Height_out;
    int numInput  = Batch * Channel * Width * Height;
    int numMask   = Map_out * Channel * K * K;

    cudaMalloc((void **) device_output_ptr, numOutput * sizeof(float));
    cudaMalloc((void **) device_input_ptr,  numInput  * sizeof(float));
    cudaMalloc((void **) device_mask_ptr,   numMask   * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, numInput * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  numMask  * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(const_mask, host_mask, numMask * sizeof(float));

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    #define OP 1
    // Set the kernel dimensions and call the kernel
    if (OP == 0) {
        dim3 dimGrid(Map_out, Height_grid * Width_grid, Batch);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    } else if (OP == 1) {
        dim3 dimGrid(Map_out, Height_grid * Width_grid, Batch);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        size_t input_tile_size = Channel * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) * sizeof(float);
        conv_forward_kernel_shared<<<dimGrid, dimBlock, input_tile_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    } else if (OP == 2) {
        dim3 dimGrid(Map_out, Height_grid * Width_grid, Batch);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, Channel);
        size_t input_tile_size = Channel * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) * sizeof(float);
        conv_forward_kernel_atomics<<<dimGrid, dimBlock, input_tile_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    } else if (OP == 3) {
        dim3 dimGrid(Map_out, Height_grid * Width_grid, Batch);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, Channel);
        size_t sum_size = Channel * TILE_WIDTH * TILE_WIDTH * sizeof(float);
        conv_forward_kernel_tree<<<dimGrid, dimBlock, sum_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    } else if (OP == 4) {

        float *X_unroll;
        cudaMalloc((void**) &X_unroll, H_unroll * W_unroll * sizeof(float));

        dim3 dimGrid1(ceil(1.0 * Channel * Height_out * Width_out / BLOCK_SIZE), 1, 1);
        dim3 dimBlock1(BLOCK_SIZE, 1, 1);
        dim3 dimGrid2(ceil(1.0 * H_unroll / TILE_WIDTH), ceil(1.0 * Map_out / TILE_WIDTH), 1);
        dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH, 1);

        for (int b = 0; b < Batch; b++) {
            unroll<<<dimGrid1, dimBlock1>>>(device_input, X_unroll, b, Channel, Height, Width, K);
            matrixMultiply<<<dimGrid2, dimBlock2>>>(X_unroll, device_output, b, Batch, Map_out, Channel, Height, Width, K);
        }
    } else if (OP == 5) {   // Multiple kernel implementations for different layer sizes
        if (Map_out == 6) {
            dim3 dimGrid(Map_out, Height_grid * Width_grid, Batch);
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
            size_t input_tile_size = Channel * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) * sizeof(float);
            conv_forward_kernel_shared<<<dimGrid, dimBlock, input_tile_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
        } else {
            dim3 dimGrid(Map_out, Height_grid * Width_grid, Batch);
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, Channel);
            size_t sum_size = Channel * TILE_WIDTH * TILE_WIDTH * sizeof(float);
            conv_forward_kernel_tree<<<dimGrid, dimBlock, sum_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);       
        }
    }


    
    #undef OP
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int numOutput = Batch * Map_out * Width_out * Height_out;
    cudaMemcpy(host_output, device_output, numOutput * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
