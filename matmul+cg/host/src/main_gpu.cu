#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include "calc_on_fpga.h"

static const int numthread = 256;

__global__ void matmul(float *a, float *b, float *c, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; // 通し番号を得るための計算
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k;
  float sum = 0.0f;
  if (i < n || j < n) {
    for(k =0; k<n; k++) {
      sum += a[i*n+k] * b[k*n+j];
    }
    c[i*n+j] = sum;
  }
}

int main(int argc, char *argv[]) {
  // check command line arguments
  ///////////////////////////////////////////
  if (argc == 1) { std::cout << "usage: ./host <name> <numdata_h> <numstream> <numtry>" << std::endl; exit(0); }
  if (argc != 5) { std::cerr << "Error! The number of arguments is wrong."              << std::endl; exit(1); }

  const char *name     = argv[1];
  const int  numdata_h = std::stoull(std::string(argv[2]));
  const int  numstream = std::stoull(std::string(argv[3]));
  const int  numtry    = std::stoull(std::string(argv[4]));
  const int  numbyte   = numdata_h * numdata_h * sizeof(float); // this sample uses "float"
  const int  numdata_d = (numdata_h/numstream);

  size_t global_item_size[3];
  size_t local_item_size[3];
  
  // host memory settings
  ///////////////////////////////////////////

  /***** GPU *****/
  float *h_a, *h_b, *h_c;
  dim3 block(64, 64), thread(16, 16);

  cudaMallocHost(&h_a, numbyte);
  cudaMallocHost(&h_b, numbyte);
  cudaMallocHost(&h_c, numbyte);
  
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      h_a[i*N+j] = (j+1)*0.0001f;
      h_b[i*N+j] = 0.5f;
      h_c[i*N+j] = 0.0f;
    }
  }

  // device memory settings
  ///////////////////////////////////////////
  float *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, numbyte);
  cudaMalloc(&d_b, numbyte);
  cudaMalloc(&d_c, numbyte);

  // main routine
  ///////////////////////////////////////////
  
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  cudaMemcpy(d_a, h_a, numbyte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, numbyte, cudaMemcpyHostToDevice);
  
  matmul<<<block, thread>>>(d_a, d_b, d_c, numdata_h);
  
  cudaMemcpy(h_c, d_c, numbyte, cudaMemcpyDeviceToHost);

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  
  // verification
  ///////////////////////////////////////////


    std::cout << std::string(30, '-') << std::endl;
    std::cout << "elapsed time: " << std::fixed << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << " sec" << std::endl;
    
  // cleanup
  ///////////////////////////////////////////
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  delete[] FPGA_calc_result;
  
  return 0;
}
