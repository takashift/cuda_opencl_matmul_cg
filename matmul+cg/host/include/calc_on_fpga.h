#ifndef CALC_ON_FPGA_H_
#define CALC_ON_FPGA_H_

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

class CalcOnFPGA {
 public:
  CalcOnFPGA();
  
  ~CalcOnFPGA();
  
  void InitOpenCL(const char *name,
                  const size_t N,
                  const size_t K,
                  const size_t VAL_SIZE,
                  size_t *global_item_size,
                  size_t *local_item_size);

  void ResetIndextoZero();

  void SendDatatoFPGA(const size_t N,
                      const size_t VAL_SIZE,
                      float *VAL,
                      int *COL_IND,
                      int *ROW_PTR,
                      float *B);
                      // aocl_utils::scoped_aligned_ptr<float> &GPU_calc_rslt_list);
  
  void Exec(const size_t *global_item_size,
            const size_t *local_item_size);
  
  void RecvDatafromFPGA(const size_t numstream,
                        float *FPGA_calc_result);
                        // aocl_utils::scoped_aligned_ptr<float> &FPGA_calc_rslt_list);
  
  void Verify(
    float* FPGA_calc_result,
    float* VAL,
    int* COL_IND,
    int* ROW_PTR,
    float* B,
    int N,
    int K,
    int VAL_SIZE);
              // aocl_utils::scoped_aligned_ptr<float> &FPGA_calc_rslt_list);

  cl_event         kernel_event;
 private:
  cl_context       context;
  cl_command_queue command_queue;
  cl_program       program;
  cl_kernel        kernel;
  cl_platform_id   platform;
  cl_event         init_event;
  cl_event         write_event[2];
  cl_event         finish_event;
  cl_mem           X_buf;  // memory object for read
  cl_mem           VAL_buf;  // memory object for read
  cl_mem           COL_IND_buf;  // memory object for read
  cl_mem           ROW_PTR_buf;  // memory object for read
  cl_mem           B_buf;  // memory object for read
};

#endif
