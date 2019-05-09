#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <cassert>
extern "C"{
  #include <bebop/smc/sparse_matrix.h>
  #include <bebop/smc/sparse_matrix_ops.h>
  #include <bebop/smc/csr_matrix.h>
}
#include "calc_on_fpga.h"

__global__ void matmul(float *a, float *b, float *c, int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; // 通し番号を得るための計算
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k;
  float sum = 0.0f;
  if (i >= N || j >= N)
    return;

  for(k=0; k<N; ++k) {
    sum += a[i*N+k] * b[k*N+j];
  }
  c[i*N+j] = sum;
}

__global__ void matrix_vector_malti(float *a,float *b, float *c, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 通し番号を得るための計算
  int j;
  float sum = 0.0;
  if (i < N) {
    for (j=0; j<N; ++j)
      sum += a[i*N+j] * b[j];
    c[i] = sum;
  }
}

void MatrixMultiplication_openmp(float *a,float *b, float *c, int N)
{
  int i, j, k;
  int chunk;
  #ifdef _OPENMP
  // omp_set_num_threads(numstream);
	if(omp_get_thread_num() == 0) {
    printf("Number of OpenMP threads %d\n", omp_get_max_threads());
    chunk = N/omp_get_max_threads();  
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


void h_matrix_vector_malti(float *a,float *b, float *c, int N)
{
  int i, j;
  int chunk;
  #ifdef _OPENMP
	if(omp_get_thread_num() == 0) {
    printf("Number of OpenMP threads %d\n", omp_get_max_threads());
    chunk = N/omp_get_max_threads();  
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

void verify_gpu(float *h_c, float *c_CPU, unsigned long N) {
  double cpu_sum = 0.0;
  double gpu_sum = 0.0;
  double rel_err = 0.0;

  #pragma omp parallel for reduction(+:cpu_sum, gpu_sum)
  for (unsigned long i=0; i<N; ++i){
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
	struct sparse_matrix_t* A_ = load_sparse_matrix(MATRIX_MARKET, "bcsstk17.mtx");
	assert(A_ != NULL);
	int errcode = sparse_matrix_convert(A_, CSR);
	if (errcode != 0)
	{
		fprintf(stderr, "*** Conversion failed! ***\n");
		// Note: Don't call destroy_sparse_matrix (A_) unless you 
		// can call free on val, ind and ptr.
		free(A_);
		exit(EXIT_FAILURE);
	}

  struct csr_matrix_t* A = (struct csr_matrix_t*) A_->repr;
  assert (A);
  assert (A->nnz == (A->rowptr[A->m] - A->rowptr[0]));
  
  // check command line arguments
  ///////////////////////////////////////////
  if (argc == 1) { std::cout << "usage: ./host <name> <numdata_h> <valsize> <numtry>"   << std::endl; exit(0); }
  if (argc != 5) { std::cerr << "Error! The number of arguments is wrong."              << std::endl; exit(1); }

  const char *name     = argv[1];
  const int  numdata_h = A->n; // std::stoull(std::string(argv[2]));
	int N = numdata_h;
  const int  valsize   = A->nnz;
	int VAL_SIZE = valsize;
  const int  numtry    = std::stoull(std::string(argv[4]));
  const unsigned long numbyte   = numdata_h * numdata_h * sizeof(float); // this sample uses "float"

  std::cout << "numdata_h: " << numdata_h << ", valsize: " << valsize << ", numtry: " << numtry << std::endl;

  size_t global_item_size[3];
  size_t local_item_size[3];
  
  // host memory settings
  ///////////////////////////////////////////

  /***** GPU *****/
  static const int numthread = 16;  
  const int numblock = (numdata_h % numthread) ? (numdata_h/numthread) + 1 : (numdata_h/numthread);
  float *h_a, *h_b, *h_c, *c_CPU, *h_vec_b, *h_vec_mul, *vec_b_CPU;

  cudaMallocHost(&h_a, numbyte);
  cudaMallocHost(&h_b, numbyte);
  cudaMallocHost(&h_c, numbyte);
  c_CPU = new float[numdata_h * numdata_h];
  vec_b_CPU = new float[numdata_h];
  cudaMallocHost(&h_vec_mul, numdata_h*sizeof(float));
  cudaMallocHost(&h_vec_b, numdata_h*sizeof(float));
  
  for (int i = 0; i < numdata_h; ++i) {
    for (int j = 0; j < numdata_h; ++j) {
      h_a[i*numdata_h+j] = (j+1)/2*0.0001f;
      h_b[i*numdata_h+j] = 0.5f;
      h_c[i*numdata_h+j] = 0.0f;
      c_CPU[i*numdata_h+j] = 0.0f;
    }
    h_vec_b[i] = 0.0f;
    h_vec_mul[i] = 0.01f; //100.0f;
    vec_b_CPU[i] = 0.0f;
  }

  /***** FPGA *****/
	int K = numtry;
  static CalcOnFPGA calc_on_fpga;
  float *FPGA_calc_result; // X_result
  float *VAL;
  int *COL_IND;
  int *ROW_PTR;
  float *B;

  posix_memalign((void **)&FPGA_calc_result, 64, N * sizeof(float));
  posix_memalign((void **)&VAL, 64, VAL_SIZE * sizeof(float));
  posix_memalign((void **)&COL_IND, 64, VAL_SIZE * sizeof(int));
  posix_memalign((void **)&ROW_PTR, 64, (N+1) * sizeof(int));
  posix_memalign((void **)&B, 64, N * sizeof(float));

  double *VAL_temp;
  posix_memalign((void **)&VAL_temp, 64, VAL_SIZE * sizeof(double));
   
  memcpy(VAL_temp, A->values, VAL_SIZE * sizeof (double));
  memcpy(COL_IND, A->colidx, VAL_SIZE * sizeof (int));
  memcpy(ROW_PTR, A->rowptr, (N+1) * sizeof (int));
  for (int i = 0; i < VAL_SIZE; ++i)
  {
        VAL[i] = (float)VAL_temp[i];
  }

  calc_on_fpga.InitOpenCL(name, N, K, VAL_SIZE, global_item_size, local_item_size);

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
  
  /***** GPU *****/
  std::chrono::system_clock::time_point start_gpu = std::chrono::system_clock::now();

  cudaMemcpy(d_a, h_a, numbyte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, numbyte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec_mul, h_vec_mul, numdata_h*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_vec_b, h_vec_b, numdata_h*sizeof(float), cudaMemcpyHostToDevice);
  
  matmul<<<dim3(numblock, numblock), dim3(numthread, numthread)>>>(d_a, d_b, d_c, numdata_h);
  matrix_vector_malti<<<dim3(numblock), dim3(numthread)>>>(d_c, d_vec_mul, d_vec_b, numdata_h);
  
  cudaMemcpy(h_vec_b, d_vec_b, numdata_h*sizeof(float), cudaMemcpyDeviceToHost);

  std::chrono::system_clock::time_point end_gpu = std::chrono::system_clock::now();
  
  cudaMemcpy(h_c, d_c, numbyte, cudaMemcpyDeviceToHost);

  /***** FPGA *****/
  for(int j=0; j<N; ++j) {
    // FPGA_calc_result[j] = 0;
		B[j] = h_vec_b[j] - VAL[j] * 1; //000000.0; // b - Ax
  }
  // ROW_PTR[N] = N;

  std::chrono::system_clock::time_point start_fpga = std::chrono::system_clock::now();

  calc_on_fpga.SendDatatoFPGA(N, VAL_SIZE, VAL, COL_IND, ROW_PTR, B);
  calc_on_fpga.Exec(global_item_size, local_item_size);  // kernel running
  // getting the computation results
  calc_on_fpga.RecvDatafromFPGA(N, FPGA_calc_result);

  std::chrono::system_clock::time_point end_fpga = std::chrono::system_clock::now();
  
  std::cout << "GPU  elapsed time: " << std::fixed << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu-start_gpu).count() << " usec" << std::endl;
  std::cout << std::string(30, '-') << std::endl;

  std::cout << "FPGA elapsed time: " << std::fixed << std::chrono::duration_cast<std::chrono::microseconds>(end_fpga-start_fpga).count() << " usec" << std::endl;
  std::cout << std::string(30, '-') << std::endl;

  // verification
  ///////////////////////////////////////////
  // MatrixMultiplication_openmp(h_a, h_b, c_CPU, numdata_h);    // 本番はコメントアウトして良い
  // h_matrix_vector_malti(c_CPU, h_vec_mul, vec_b_CPU, numdata_h);    // 本番はコメントアウトして良い

  // verify_gpu(h_c, c_CPU, numdata_h*numdata_h); // 行列積チェック
  verify_gpu(h_vec_b, vec_b_CPU, numdata_h); // d_vec_b チェック

  calc_on_fpga.Verify(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);
    
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

  // destroy_sparse_matrix(A_);
  // destroy_csr_matrix(A);

  delete[] FPGA_calc_result;
  
  return 0;
}
