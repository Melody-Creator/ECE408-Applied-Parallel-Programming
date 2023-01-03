// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_2D 16
#define BLOCK_SIZE_SCAN 128

//@@ insert code here
__global__ void castImage(float *input, unsigned char *ucharImage, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)  ucharImage[i] = (unsigned char) (255 * input[i]);
}

__global__ void castImageBack(float *output, unsigned char *ucharImage, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)  output[i] = (float) (ucharImage[i] / 255.0);
}

__global__ void convertImage(unsigned char *grayImage, unsigned char *ucharImage, int height, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int idx = row * width + col;
    unsigned char r = ucharImage[3 * idx];
    unsigned char g = ucharImage[3 * idx + 1];
    unsigned char b = ucharImage[3 * idx + 2];
    grayImage[idx] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void computeHisto(unsigned char *grayImage, unsigned int *histo, int n) {
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < HISTOGRAM_LENGTH)  histo_private[threadIdx.x] = 0;

  __syncthreads();
  if (i < n)  atomicAdd(histo_private + grayImage[i], 1);

  __syncthreads();
  if (threadIdx.x < HISTOGRAM_LENGTH)  atomicAdd(histo + threadIdx.x, histo_private[threadIdx.x]);
}

__global__ void scan(unsigned int *input, float *output, int len, int div) {
  __shared__ float T[BLOCK_SIZE_SCAN << 1];

  int t = threadIdx.x, start = 2 * blockIdx.x * blockDim.x;
  T[t] = (start + t < len) ? 1.0 * input[start + t] : 0.0;
  T[blockDim.x + t] = (start + blockDim.x + t < len) ? 1.0 * input[start + blockDim.x + t] : 0.0;

  for (int stride = 1; stride <= blockDim.x; stride <<= 1) {
    __syncthreads();
    int index = (t + 1) * (stride << 1) - 1;
    if (index < (BLOCK_SIZE_SCAN << 1) && index - stride >= 0)  
      T[index] += T[index - stride];
  }

  for (int stride = (BLOCK_SIZE_SCAN >> 1); stride >= 1; stride >>= 1) {
    __syncthreads();
    int index = (t + 1) * (stride << 1) - 1;
    if (index + stride < (BLOCK_SIZE_SCAN << 1))  
      T[index + stride] += T[index];
  }

  __syncthreads();
  if (start + t < len)  output[start + t] = T[t] / div;
  if (start + blockDim.x + t < len)  output[start + blockDim.x + t] = T[blockDim.x + t] / div;
}

__global__ void equalize(float *cdf, unsigned char *ucharImage, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned char val = ucharImage[i];
    ucharImage[i] = min(max(255.0 * (cdf[val] - cdf[0]) / (1.0 - cdf[0]), 0.0), 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInput;
  float *hostOutput;
  const char *inputImageFile;

  //@@ Insert more code here
  int numPixels;
  float *deviceInput;
  float *deviceOutput;
  float *cdf;
  unsigned char *ucharImage;
  unsigned char *grayImage;
  unsigned int *histo;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInput = wbImage_getData(inputImage);
  hostOutput = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  numPixels = imageWidth * imageHeight;
  cudaMalloc((void **) &deviceInput,  numPixels * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutput, numPixels * imageChannels * sizeof(float));

  cudaMalloc((void **) &ucharImage, numPixels * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &grayImage,  numPixels * sizeof(unsigned char));

  cudaMalloc((void **) &histo,   HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void  *) histo, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &cdf,     HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(deviceInput, hostInput, numPixels * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid;
  dim3 dimBlock;

  // castImage
  dimGrid  = dim3(ceil(1.0*numPixels*imageChannels/BLOCK_SIZE_1D), 1, 1);
  dimBlock = dim3(BLOCK_SIZE_1D, 1, 1);

  castImage<<<dimGrid, dimBlock>>>(deviceInput, ucharImage, numPixels * imageChannels);
  cudaDeviceSynchronize();

  // convertImage
  dimGrid  = dim3(ceil(1.0*imageWidth/BLOCK_SIZE_2D), ceil(1.0*imageHeight/BLOCK_SIZE_2D), 1);
  dimBlock = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);

  convertImage<<<dimGrid, dimBlock>>>(grayImage, ucharImage, imageHeight, imageWidth);
  cudaDeviceSynchronize();

  // computeHisto
  dimGrid  = dim3(ceil(1.0*numPixels/BLOCK_SIZE_1D), 1, 1);
  dimBlock = dim3(BLOCK_SIZE_1D, 1, 1);

  computeHisto<<<dimGrid, dimBlock>>>(grayImage, histo, numPixels);
  cudaDeviceSynchronize();

  // scan
  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(BLOCK_SIZE_SCAN, 1, 1);

  scan<<<dimGrid, dimBlock>>>(histo, cdf, HISTOGRAM_LENGTH, numPixels);
  cudaDeviceSynchronize();

  // equalize
  dimGrid  = dim3(ceil(1.0*numPixels*imageChannels/BLOCK_SIZE_1D), 1, 1);
  dimBlock = dim3(BLOCK_SIZE_1D, 1, 1);

  equalize<<<dimGrid, dimBlock>>>(cdf, ucharImage, numPixels * imageChannels);
  cudaDeviceSynchronize();

  // castImageBack
  dimGrid  = dim3(ceil(1.0*numPixels*imageChannels/BLOCK_SIZE_1D), 1, 1);
  dimBlock = dim3(BLOCK_SIZE_1D, 1, 1);

  castImageBack<<<dimGrid, dimBlock>>>(deviceOutput, ucharImage, numPixels * imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutput, deviceOutput, numPixels * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbImage_setData(outputImage, hostOutput);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(ucharImage);
  cudaFree(grayImage);
  cudaFree(histo);
  cudaFree(cdf);

  free(hostInput);
  free(hostOutput);

  return 0;
}
