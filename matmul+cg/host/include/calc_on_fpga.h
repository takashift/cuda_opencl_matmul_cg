/******************************************************************************/
/* A sample program for reducing GPU computation results                      */
/*                                                Ryohei Kobayashi 2018.06.12 */
/******************************************************************************/
#ifndef CALC_ON_FPGA_H_
#define CALC_ON_FPGA_H_

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

class CalcOnFPGA {
 public:
  CalcOnFPGA();
  
  ~CalcOnFPGA();
  
  void InitOpenCL(const char *name,
                  const float *X,
                  const float *VAL,
                  const int *COL_IND,
                  const int *ROW_PTR,
                  const float *B,
                  const int N,
                  const int K,
                  const int VAL_SIZE,
                  size_t *global_item_size,
                  size_t *local_item_size);

  void ResetIndextoZero();

  void SendDatatoFPGA(const size_t numdata_d,
                      const int stm,
                      float *GPU_calc_rslt_list);
                      // aocl_utils::scoped_aligned_ptr<float> &GPU_calc_rslt_list);
  
  void Exec(const size_t *global_item_size,
            const size_t *local_item_size);
  
  void RecvDatafromFPGA(const size_t numstream,
                        float *FPGA_calc_rslt_list);
                        // aocl_utils::scoped_aligned_ptr<float> &FPGA_calc_rslt_list);
  
  void Verify(const size_t numdata_d, 
              const size_t numstream,
              const size_t numtry, 
              float *FPGA_calc_rslt_list);
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
  cl_mem           COL_buf;  // memory object for read
  cl_mem           ROW_buf;  // memory object for read
  cl_mem           B_buf;  // memory object for read
};

#endif
