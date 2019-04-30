#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include "calc_on_fpga.h"

// static const int numthread = 256;

// __global__
// void vecadd(float *a, float *b, float *c, int n) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   int j = blockIdx.y * blockDim.y + threadIdx.y;
//   if (i < n || j < n) {
//     c[i*n+j] += a[i*n+k] * b[k*n+j];
//   }
// }

int main(int argc, char *argv[]) {
  // check command line arguments
  ///////////////////////////////////////////
  if (argc == 1) { std::cout << "usage: ./host <name> <numdata_h> <numstream> <numtry>" << std::endl; exit(0); }
  if (argc != 5) { std::cerr << "Error! The number of arguments is wrong."              << std::endl; exit(1); }

  const char *name     = argv[1];
  const int  numdata_h = std::stoull(std::string(argv[2]));
  const int  numstream = std::stoull(std::string(argv[3]));
  const int  numtry    = std::stoull(std::string(argv[4]));
  const int  numbyte   = numdata_h * sizeof(float); // this sample uses "float"
  const int  numdata_d = (numdata_h/numstream);

  size_t global_item_size[3];
  size_t local_item_size[3];
  
  // host memory settings
  ///////////////////////////////////////////

  /***** FPGA *****/
  static CalcOnFPGA calc_on_fpga;
  int N = 1000;
  int K = 1000;
  int VAL_SIZE = 1000;
  float *FPGA_calc_result = new float[N];
  float *VAL;
  int *COL_IND;
  int *ROW_PTR;
  float *B;

  posix_memalign((void **)&VAL, 64, VAL_SIZE * sizeof(double));
  posix_memalign((void **)&COL_IND, 64, VAL_SIZE * sizeof(double));
  posix_memalign((void **)&ROW_PTR, 64, N * sizeof(double));
  posix_memalign((void **)&B, 64, N * sizeof(double));

  for(int i=0; i<VAL_SIZE; i++) {
    VAL[i] = i+1;
    COL_IND[i] = i;
  }
  for(int j=0; j<N; j++) {
    // FPGA_calc_result[j] = 0;
    ROW_PTR[j] = j*N+j;
    B[j] = j/2 - 0.0; // x=0.0; b - Ax
  }

  calc_on_fpga.InitOpenCL(name, N, K, VAL_SIZE, global_item_size, local_item_size);

  // main routine
  ///////////////////////////////////////////
  // const int numblock = (numdata_h % numthread) ? (numdata_h/numthread) + 1 : (numdata_h/numthread);
  
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  // clWaitForEvents(1, &calc_on_fpga.kernel_event);
  calc_on_fpga.SendDatatoFPGA(N, VAL_SIZE, VAL, COL_IND, ROW_PTR, B);
  calc_on_fpga.Exec(global_item_size, local_item_size);  // kernel running
  // getting the computation results
  calc_on_fpga.RecvDatafromFPGA(N, FPGA_calc_result);

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  
  // verification
  ///////////////////////////////////////////


    std::cout << std::string(30, '-') << std::endl;
    std::cout << "elapsed time: " << std::fixed << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << " sec" << std::endl;
    
  // cleanup
  ///////////////////////////////////////////
  // cudaFreeHost(h_a);
  // cudaFreeHost(h_b);
  // cudaFreeHost(h_c);
  // cudaFree(d_a);
  // cudaFree(d_b);
  // cudaFree(d_c);
  // for (int stm = 0; stm < numstream; ++stm) {
  //   cudaStreamDestroy(stream[stm]);
  // }

  delete[] FPGA_calc_result;
  
  return 0;
}
