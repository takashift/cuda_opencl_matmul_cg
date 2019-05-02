#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <omp.h>
#include <chrono>
// #include "calc_on_fpga.h"

__global__ void matmul(float *a, float *b, float *c, unsigned long n) {
  unsigned long j = blockIdx.x * blockDim.x + threadIdx.x; // 通し番号を得るための計算
  unsigned long i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned long k;
  float sum = 0.0f;
  if (i >= n || j >= n)
    return;

  for(k=0; k<n; ++k) {
    sum += a[i*n+k] * b[k*n+j];
  }
  c[i*n+j] = sum;
}

__global__ void matrix_vector_malti(float *a,float *b, float *c, unsigned long N)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x; // 通し番号を得るための計算
  unsigned long j;
  float sum = 0.0;
  if (i < N) {
    for (j=0; j<N; ++j)
      sum += a[i*N+j] * b[j];
    c[i] = sum;
  }
}

void MatrixMultiplication_openmp(float *a,float *b, float *c, unsigned long N)
{
  unsigned long i, j, k;
  int chunk;
  #ifdef _OPENMP
  // omp_set_num_threads(numstream);
	if(omp_get_thread_num() == 0) {
    printf("Number of OpenMP threads %d\n", omp_get_num_threads());
    chunk = N/omp_get_num_threads();  
	}
  #endif

#pragma omp parallel shared(a,b,c,chunk) private(i,j,k)
  {
#pragma omp for
    for (i=0; i<N; ++i){
      for (j=0; j<N; ++j){
        float sum = 0.0 ;
        for (k=0; k<N; ++k)
          sum += a[i*N+k] * b[k*N+j];
        c[i*N+j] = sum;
      }
    }
  }
}


void h_matrix_vector_malti(float *a,float *b, float *c, unsigned long N)
{
  unsigned long i, j;
  int chunk;
  #ifdef _OPENMP
	if(omp_get_thread_num() == 0) {
    printf("Number of OpenMP threads %d\n", omp_get_num_threads());
    chunk = N/omp_get_num_threads();  
	}
  #endif

#pragma omp parallel shared(a,b,c,chunk) private(i,j)
  {
#pragma omp for
    for (i=0; i<N; ++i){
      float sum = 0.0 ;
      for (j=0; j<N; ++j)
        sum += a[i*N+j]*b[j];
      c[i] = sum;
    }
  }
}

void verify(float *h_c, float *c_CPU, unsigned long numdata_h) {
  double cpu_sum = 0.0;
  double gpu_sum = 0.0;
  double rel_err = 0.0;

  #pragma omp parallel for reduction(+:cpu_sum, gpu_sum)
  // for (unsigned long i=0; i<numdata_h*numdata_h; ++i){
  for (unsigned long i=0; i<numdata_h; ++i){  // d_vec_b チェック
    // std::cout << c_CPU[i] << "(CPU) " << std::endl;
    // std::cout << h_c[i] << "(GPU) " << std::endl;
    cpu_sum += (double)c_CPU[i]*c_CPU[i];
    gpu_sum += (double)h_c[i]*h_c[i];
  }

  cpu_sum = sqrt(cpu_sum);
  gpu_sum = sqrt(gpu_sum);
  if( cpu_sum > gpu_sum ) {
    rel_err = (cpu_sum-gpu_sum)/cpu_sum;
  } else {
    rel_err = (gpu_sum-cpu_sum)/cpu_sum;
  }

  if(rel_err < 1e-6)
  {
      printf("Verification Successful err = %e\n", rel_err);
  }
  else
  {
      printf("Verification Fail err = %e\n", rel_err);
  }
  printf("ResultGPU = %lf\n", gpu_sum);
  printf("ResultCPU = %lf\n", cpu_sum);
} 

int main(int argc, char *argv[]) {
  // check command line arguments
  ///////////////////////////////////////////
  if (argc == 1) { std::cout << "usage: ./host <name> <numdata_h> <numstream> <numtry>" << std::endl; exit(0); }
  if (argc != 5) { std::cerr << "Error! The number of arguments is wrong."              << std::endl; exit(1); }

  const char *name     = argv[1];
  const unsigned long numdata_h = std::stoull(std::string(argv[2]));
  const int  numstream = std::stoull(std::string(argv[3]));
  const int  numtry    = std::stoull(std::string(argv[4]));
  const unsigned long numbyte   = numdata_h * numdata_h * sizeof(float); // this sample uses "float"

  // size_t global_item_size[3];
  // size_t local_item_size[3];
  
  // host memory settings
  ///////////////////////////////////////////

  /***** GPU *****/
  static const int numthread = 16;  
  const unsigned long numblock = (numdata_h % numthread) ? (numdata_h/numthread) + 1 : (numdata_h/numthread);
  float *h_a, *h_b, *h_c, *c_CPU, *h_vec_b, *h_vec_mul, *vec_b_CPU;

  cudaMallocHost(&h_a, numbyte);
  cudaMallocHost(&h_b, numbyte);
  cudaMallocHost(&h_c, numbyte);
  c_CPU = new float[numdata_h * numdata_h];
  vec_b_CPU = new float[numdata_h * numdata_h];
  cudaMallocHost(&h_vec_mul, numdata_h*sizeof(float)); // h_vec_mul = new float[numdata_h];
  cudaMallocHost(&h_vec_b, numdata_h*sizeof(float)); // h_vec_b = new float[];
  
  for (unsigned long i = 0; i < numdata_h; ++i) {
    for (unsigned long j = 0; j < numdata_h; ++j) {
      h_a[i*numdata_h+j] = (j+1)/2*0.0001f;
      h_b[i*numdata_h+j] = 0.5f;
      h_c[i*numdata_h+j] = 0.0f;
      c_CPU[i*numdata_h+j] = 0.0f;
    }
    h_vec_b[i] = 0.0f;
    h_vec_mul[i] = 0.01f;
    vec_b_CPU[i] = 0.0f;
  }

  // device memory settings
  ///////////////////////////////////////////
  float *d_a, *d_b, *d_c, *d_vec_mul, *d_vec_b;

  cudaMalloc(&d_a, numbyte);
  cudaMalloc(&d_b, numbyte);
  cudaMalloc(&d_c, numbyte);
  cudaMalloc(&d_vec_mul, numdata_h*sizeof(float));
  cudaMalloc(&d_vec_b, numdata_h*sizeof(float));
 
  // main routine
  ///////////////////////////////////////////
  
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  cudaMemcpy(d_a, h_a, numbyte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, numbyte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec_mul, h_vec_mul, numdata_h*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_vec_b, h_vec_b, numdata_h*sizeof(float), cudaMemcpyHostToDevice);
  
  matmul<<<dim3(numblock, numblock), dim3(numthread, numthread)>>>(d_a, d_b, d_c, numdata_h);
  matrix_vector_malti<<<dim3(numblock), dim3(numthread)>>>(d_c, d_vec_mul, d_vec_b, numdata_h);
  
  cudaMemcpy(h_vec_b, d_vec_b, numdata_h*sizeof(float), cudaMemcpyDeviceToHost);

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  
  cudaMemcpy(h_c, d_c, numbyte, cudaMemcpyDeviceToHost);


  // verification
  ///////////////////////////////////////////
  MatrixMultiplication_openmp(h_a, h_b, c_CPU, numdata_h);
  h_matrix_vector_malti(c_CPU, h_vec_mul, vec_b_CPU, numdata_h);

  // verify(h_c, c_CPU, numdata_h);
  verify(h_vec_b, vec_b_CPU, numdata_h);

    std::cout << std::string(30, '-') << std::endl;
    std::cout << "elapsed time: " << std::fixed << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " usec" << std::endl;
    
  // cleanup
  ///////////////////////////////////////////
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFreeHost(h_vec_mul);
  cudaFreeHost(h_vec_b);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_vec_b);
  cudaFree(d_vec_b);

  // delete[] FPGA_calc_result;
  
  return 0;
}
