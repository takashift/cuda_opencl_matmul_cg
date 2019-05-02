#include "calc_on_fpga.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <chrono>
// #include <fenv.h>

/********************************************************************/
CalcOnFPGA::CalcOnFPGA() {
  context       = NULL;
  command_queue = NULL;
  program       = NULL;
  kernel        = NULL;
  platform      = NULL;
}

/********************************************************************/
CalcOnFPGA::~CalcOnFPGA() {
  clReleaseEvent(init_event);
  clReleaseEvent(kernel_event);
  clReleaseEvent(finish_event);
  for (int i = 0; i < 2; ++i) {
    clReleaseEvent(write_event[i]);
  }
  clFlush(command_queue);
  clFinish(command_queue);
  clReleaseMemObject(X_buf);
  clReleaseMemObject(VAL_buf);
  clReleaseMemObject(COL_IND_buf);
  clReleaseMemObject(ROW_PTR_buf);
  clReleaseMemObject(B_buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
}

/********************************************************************/
void CalcOnFPGA::InitOpenCL(const char   *name,
                            const size_t N,
                            const size_t K,
                            const size_t VAL_SIZE,
                            size_t       *global_item_size,
                            size_t       *local_item_size) {

  cl_int                                 status;
  cl_uint                                num_devices = 0;
  aocl_utils::scoped_array<cl_device_id> device_id;

  // work item
  local_item_size[2] = 1;
  local_item_size[1] = 1;
  local_item_size[0] = 1;
  global_item_size[2] = 1;
  global_item_size[1] = 1;
  global_item_size[0] = 1;
  
  std::cout << "Initializing OpenCL" << std::endl;

  if (!aocl_utils::setCwdToExeDir()) exit(1);

  // Get the OpenCL platform.
  platform = aocl_utils::findPlatform("Intel(R) FPGA");  // ~ 16.0: aocl_utils::findPlatform("Altera");
  if (platform == NULL) {
    std::cerr << "ERROR: Unable to find Intel(R) FPGA OpenCL platform." << std::endl;
    exit(1);
  }

  // Query the available OpenCL device.
  device_id.reset(aocl_utils::getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  std::cout << "Platform: " << aocl_utils::getPlatformName(platform).c_str() << std::endl;
  std::cout << "Using " << num_devices << " device(s)" << std::endl;
  std::cout << " " << aocl_utils::getDeviceName(device_id[0]).c_str() << std::endl;
  
  // Create the context.
  context = clCreateContext(NULL, num_devices, device_id, NULL, NULL, &status);
  aocl_utils::checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = aocl_utils::getBoardBinaryFile(name, device_id[0]);
  std::cout << "Using AOCX: " << binary_file.c_str() << std::endl;
  program = aocl_utils::createProgramFromBinary(context, binary_file.c_str(), device_id, num_devices);
  
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  aocl_utils::checkError(status, "Failed to build program");

  // kernel
  kernel = clCreateKernel(program, name, &status);
  if (status != CL_SUCCESS) {
    std::cerr << "clCreateKernel() error" << std::endl;
    exit(1);
  }

  // command queue
  command_queue = clCreateCommandQueue(context, device_id[0], 0, &status);

  // memory object_m
  X_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*N, NULL, &status);
  aocl_utils::checkError(status, "Failed to create buffer for X");
  VAL_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*VAL_SIZE, NULL, &status);
  aocl_utils::checkError(status, "Failed to create buffer for VAL");
  COL_IND_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*VAL_SIZE, NULL, &status);
  aocl_utils::checkError(status, "Failed to create buffer for COL_IND");
  ROW_PTR_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*(N+1), NULL, &status);
  aocl_utils::checkError(status, "Failed to create buffer for ROW_PTR");
  B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*N, NULL, &status);
  aocl_utils::checkError(status, "Failed to create buffer for B");

  // Set kernel arguments.
  unsigned argi = 0;
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &X_buf);       aocl_utils::checkError(status, "Failed to set argument X");
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &VAL_buf);     aocl_utils::checkError(status, "Failed to set argument VAL");
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &COL_IND_buf);     aocl_utils::checkError(status, "Failed to set argument COL");
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &ROW_PTR_buf);     aocl_utils::checkError(status, "Failed to set argument ROW");
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &B_buf);       aocl_utils::checkError(status, "Failed to set argument B");
  status = clSetKernelArg(kernel, argi++, sizeof(int),    &N);           aocl_utils::checkError(status, "Failed to set argument N");
  status = clSetKernelArg(kernel, argi++, sizeof(int),    &K);           aocl_utils::checkError(status, "Failed to set argument K");
  status = clSetKernelArg(kernel, argi++, sizeof(int),    &VAL_SIZE);    aocl_utils::checkError(status, "Failed to set argument VAL_SIZE");
}

/********************************************************************/
void CalcOnFPGA::SendDatatoFPGA(const size_t N,
                                const size_t VAL_SIZE,
                                float *VAL,
                                int *COL_IND,
                                int *ROW_PTR,
                                float *B) {
  // host to device_m
  cl_int status;
  status = clEnqueueWriteBuffer(command_queue, VAL_buf, CL_TRUE, 0, sizeof(float)*VAL_SIZE, VAL, 0, NULL, NULL);
  aocl_utils::checkError(status, "Failed to transfer input VAL");
  status = clEnqueueWriteBuffer(command_queue, COL_IND_buf, CL_TRUE, 0, sizeof(int)*VAL_SIZE, COL_IND, 0, NULL, NULL);
  aocl_utils::checkError(status, "Failed to transfer input COL_IND");
  status = clEnqueueWriteBuffer(command_queue, ROW_PTR_buf, CL_TRUE, 0, sizeof(int)*(N+1), ROW_PTR, 0, NULL, NULL);
  aocl_utils::checkError(status, "Failed to transfer input ROW_PTR");
  status = clEnqueueWriteBuffer(command_queue, B_buf, CL_TRUE, 0, sizeof(float)*N, B, 0, NULL, &write_event[0]);
  aocl_utils::checkError(status, "Failed to transfer input B");
}


/********************************************************************/
void CalcOnFPGA::Exec(const size_t *global_item_size,
                      const size_t *local_item_size) {
  // Kernel kicked
  cl_int status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_item_size, local_item_size, 1, &write_event[0], &kernel_event);
  aocl_utils::checkError(status, "Failed to launch kernel");
}


/********************************************************************/
void CalcOnFPGA::RecvDatafromFPGA(const size_t numstream,
                                  float        *FPGA_calc_result) {
  // device to host_m
  cl_int status = clEnqueueReadBuffer(command_queue, X_buf, CL_TRUE, 0, sizeof(float)*numstream, FPGA_calc_result, 1, &kernel_event, &finish_event);
  aocl_utils::checkError(status, "Failed to transfer output X");
}


/********************************************************************/
void CalcOnFPGA::Verify(
    float* FPGA_calc_result,
    float* VAL,
    int* COL_IND,
    int* ROW_PTR,
    float* B,
    int N,
    int K,
    int VAL_SIZE
	  )
{
	// float *x = new float[N], *r = new float[N], *p = new float[N], *y = new float[N], alfa, beta;
	// float *VAL_local = new float[VAL_SIZE];
	// int *COL_IND_local = new int[VAL_SIZE], *ROW_PTR_local = new int[N + 1];
	// float temp_sum, temp_pap, temp_rr1, temp_rr2;
  int error = N;

	float x[N], r[N], p[N], y[N], alfa, beta;
	float VAL_local[VAL_SIZE];
	int COL_IND_local[VAL_SIZE], ROW_PTR_local[N + 1];
	float temp_sum, temp_pap, temp_rr1, temp_rr2, sum = 0;

  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	temp_rr1 = 0.0f;
	for(int i = 0; i < N; ++i){
		ROW_PTR_local[i] = ROW_PTR[i];
		x[i] = 0.0f;
		r[i] = B[i];
		p[i] = B[i];
		temp_rr1 += r[i] * r[i];
	}
	ROW_PTR_local[N] = ROW_PTR[N];

	for(int i = 0; i < VAL_SIZE; ++i){
		COL_IND_local[i] = COL_IND[i];
		VAL_local[i] = VAL[i];
	}

	for(int i = 0; i < K; ++i){
		temp_pap = 0.0f;
		for(int j = 0; j < N; ++j){
			temp_sum = 0.0f;
			for(int l = ROW_PTR_local[j]; l < ROW_PTR_local[j + 1]; l++){
				temp_sum += p[COL_IND_local[l]] * VAL_local[l];
			}
			y[j] = temp_sum;
			temp_pap += p[j] * temp_sum;
		}

		alfa = temp_rr1 / temp_pap;

		temp_rr2 = 0.0f;
		for(int j = 0; j < N; ++j){
			x[j] += alfa * p[j];
			r[j] -= alfa * y[j];
			temp_rr2 += r[j] * r[j];
		}

		beta = temp_rr2 / temp_rr1;

		for(int j = 0; j < N; ++j){
			p[j] = r[j] + beta * p[j];
		}
		temp_rr1 = temp_rr2;

	}

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

// if (fetestexcept(FE_INVALID)) {
//    puts("浮動小数点例外が発生しました");
// }
// if (fetestexcept(FE_DIVBYZERO)) {
//    puts("ゼロ除算が発生しました");
// }
// if (fetestexcept(FE_OVERFLOW)) {
//    puts("オーバーフローが発生しました");
// }
// if (fetestexcept(FE_UNDERFLOW)) {
//    puts("アンダーフローが発生しました");
// }
// if (fetestexcept(FE_INEXACT)) {
//    puts("不正確な結果が発生しました");
// }

	for(int j = 0; j < N; ++j){
    // std::cout << "FPGA" << FPGA_calc_result[j] << ", CPU"<< x[j] << std::endl;
		if(FPGA_calc_result[j] != x[j]) {
      error = j;
      break;
    }
    sum += FPGA_calc_result[j];
	}

  if (error == N) {
    std::cout << std::string(30, '-') << std::endl;
    std::cout << "FPGA Verification: PASS" << std::endl;
    std::cout << "ResultFPGA = " << sum << std::endl;
  } else {
    std::cout << "Error! FPGA Verification failed..." << error << std::endl;
  }
  std::cout << "elapsed time: " << std::fixed << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " usec" << std::endl;
}