#include "calc_on_fpga.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>


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
  ROW_PTR_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*N, NULL, &status);
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
  cl_int status = clEnqueueWriteBuffer(command_queue, VAL_buf, CL_FALSE, 0, sizeof(float)*VAL_SIZE, VAL, 1, &init_event, &write_event[0]);
  aocl_utils::checkError(status, "Failed to transfer input VAL");
  cl_int status = clEnqueueWriteBuffer(command_queue, COL_IND_buf, CL_FALSE, 0, sizeof(float)*VAL_SIZE, COL_IND, 1, &init_event, &write_event[0]);
  aocl_utils::checkError(status, "Failed to transfer input COL_IND");
  cl_int status = clEnqueueWriteBuffer(command_queue, ROW_PTR_buf, CL_FALSE, 0, sizeof(float)*N, ROW_PTR, 1, &init_event, &write_event[0]);
  aocl_utils::checkError(status, "Failed to transfer input ROW_PTR");
  cl_int status = clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0, sizeof(float)*N, B, 1, &init_event, &write_event[0]);
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
void CalcOnFPGA::Verify(const size_t numdata_d,
                        const size_t numstream,
                        const size_t numtry, 
                        float        *FPGA_calc_result) {

  std::cout << std::endl;

  // error check
  bool error = false;
#pragma omp parallel for
  for (size_t i = 0; i < numstream; ++i) {
    const int   thread_id               = omp_get_thread_num();
    const float c0                      = 3.0f + static_cast<float>(thread_id * 2);
    const float expected_reduction_rslt = std::pow((2*1+1), (numtry-1)) * c0 * numdata_d;
    if (FPGA_calc_result[i] != expected_reduction_rslt) error = true;
  }

  if (!error) {
    std::cout << std::string(30, '-') << std::endl;
    std::cout << "FPGA Verification: PASS" << std::endl;
  } else {
    std::cout << "Error! FPGA Verification failed..." << std::endl;
    for (size_t i = 0; i < numstream; ++i) {
      const float c0                      = 3.0f + static_cast<float>(i * 2);
      const float expected_reduction_rslt = std::pow((2*1+1), (numtry-1)) * c0 * numdata_d;
      if (FPGA_calc_result[i] != expected_reduction_rslt) {
        std::cout << "FPGA_calc_result[" << i << "]: " << std::fixed << FPGA_calc_result[i];
        std::cout << ", expected: " << expected_reduction_rslt << std::endl;
        break;
      }
    }
  }
}
