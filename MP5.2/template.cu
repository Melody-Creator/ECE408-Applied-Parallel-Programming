// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *aux, int flag) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[BLOCK_SIZE << 1];
  int t = threadIdx.x, start = 2 * blockIdx.x * blockDim.x;
  T[t] = (start + t < len) ?  input[start + t] : 0.0;
  T[blockDim.x + t] = (start + blockDim.x + t < len) ? input[start + blockDim.x + t] : 0.0;

  for (int stride = 1; stride <= blockDim.x; stride <<= 1) {
    __syncthreads();
    int index = (t + 1) * (stride << 1) - 1;
    if (index < (BLOCK_SIZE << 1) && index - stride >= 0)  
      T[index] += T[index - stride];
  }

  for (int stride = (BLOCK_SIZE >> 1); stride >= 1; stride >>= 1) {
    __syncthreads();
    int index = (t + 1) * (stride << 1) - 1;
    if (index + stride < (BLOCK_SIZE << 1))  
      T[index + stride] += T[index];
  }

  __syncthreads();
  if (start + t < len)  output[start + t] = T[t];
  if (start + blockDim.x + t < len)  output[start + blockDim.x + t] = T[blockDim.x + t];
  if (flag == 1 && t == BLOCK_SIZE - 1)  
    aux[blockIdx.x] = T[(BLOCK_SIZE << 1) - 1];
}

__global__ void add(float *output, float *aux, int len) {
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x != 0 && i < len)  output[i] += aux[blockIdx.x - 1];
  if (blockIdx.x != 0 && i + blockDim.x < len)  output[i + blockDim.x] += aux[blockIdx.x - 1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *AuxArray;
  int numElements; // number of elements in the list
  int numBlocks;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  numBlocks = (numElements - 1) / (BLOCK_SIZE << 1) + 1;
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&AuxArray, numBlocks * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid1(numBlocks, 1, 1);
  dim3 dimGrid2(1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  scan<<<dimGrid1, dimBlock>>>(deviceInput, deviceOutput, numElements, AuxArray, 1);
  scan<<<dimGrid2, dimBlock>>>(AuxArray, AuxArray, numBlocks, AuxArray, 0);
  add<<<dimGrid1, dimBlock>>>(deviceOutput, AuxArray, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
