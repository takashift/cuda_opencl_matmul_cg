#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include "calc_on_fpga.h"

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
  float *FPGA_calc_result; // = new float[N];
  float *VAL;
  int *COL_IND;
  int *ROW_PTR;
  float *B;

  posix_memalign((void **)&FPGA_calc_result, 64, N * sizeof(float));
  posix_memalign((void **)&VAL, 64, VAL_SIZE * sizeof(float));
  posix_memalign((void **)&COL_IND, 64, VAL_SIZE * sizeof(int));
  posix_memalign((void **)&ROW_PTR, 64, N+1 * sizeof(int));
  posix_memalign((void **)&B, 64, N * sizeof(float));

  for(int i=0; i<VAL_SIZE; i++) {
    VAL[i] = i * 50000.0f;
    COL_IND[i] = i;
  }
  for(int j=0; j<N; j++) {
    // FPGA_calc_result[j] = 0;
    ROW_PTR[j] = j;
    B[j] = -j - VAL[j] * 2.0f; // b - Ax
  }
  ROW_PTR[N] = N;

  calc_on_fpga.InitOpenCL(name, N, K, VAL_SIZE, global_item_size, local_item_size);

  // main routine
  ///////////////////////////////////////////
  
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  calc_on_fpga.SendDatatoFPGA(N, VAL_SIZE, VAL, COL_IND, ROW_PTR, B);
  calc_on_fpga.Exec(global_item_size, local_item_size);  // kernel running
  // getting the computation results
  calc_on_fpga.RecvDatafromFPGA(N, FPGA_calc_result);

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  
  std::cout << std::string(30, '-') << std::endl;
  std::cout << "elapsed time: " << std::fixed << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " usec" << std::endl;

  // verification
  ///////////////////////////////////////////
  // calc_on_fpga.Verify(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);
    
  // cleanup
  ///////////////////////////////////////////


  delete[] FPGA_calc_result;
  
  return 0;
}
