#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
const int TILE_WIDTH = 8;
const int KERNEL_SIZE = 3;

//@@ Define constant memory for device kernel here
__constant__ float kernel[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float tile[TILE_WIDTH + 2][TILE_WIDTH + 2][TILE_WIDTH + 2];

  int tz = threadIdx.z, ty = threadIdx.y, tx = threadIdx.x;
  int outputZ = blockIdx.z * TILE_WIDTH + tz;
  int outputY = blockIdx.y * TILE_WIDTH + ty;
  int outputX = blockIdx.x * TILE_WIDTH + tx;
  int inputZ = outputZ - 1, inputY = outputY - 1, inputX = outputX - 1;

  tile[tz][ty][tx] = (inputZ >= 0 && inputZ < z_size) && (inputY >= 0 && inputY < y_size) && 
                     (inputX >= 0 && inputX < x_size) ? input[inputZ * y_size * x_size + 
                      inputY * x_size + inputX] : 0.0;
  __syncthreads();

  float val = 0.0;
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH) {
    for (int i = 0; i < KERNEL_SIZE; i++)
      for (int j = 0; j < KERNEL_SIZE; j++) 
        for (int k = 0; k < KERNEL_SIZE; k++) 
          val += kernel[i][j][k] * tile[i+tz][j+ty][k+tx];
    if (outputZ < z_size && outputY < y_size && outputX < x_size)  
      output[outputZ * y_size * x_size + outputY * x_size + outputX] = val;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput,  (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel, hostKernel, kernelLength * sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(1.0*x_size/TILE_WIDTH), ceil(1.0*y_size/TILE_WIDTH), ceil(1.0*z_size/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH + 2, TILE_WIDTH + 2, TILE_WIDTH + 2);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
