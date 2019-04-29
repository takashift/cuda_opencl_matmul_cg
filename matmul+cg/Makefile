# Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 
# This agreement shall be governed in all respects by the laws of the State of California and
# by the laws of the United States of America.
# This is a GNU Makefile.

# You must configure ALTERAOCLSDKROOT to point the root directory of the Altera SDK for OpenCL
# software installation.
# See http://www.altera.com/literature/hb/opencl-sdk/aocl_getting_started.pdf 
# for more information on installing and configuring the Altera SDK for OpenCL.


# Settings of the host code compilation
########################################################################
# Where is the Altera SDK for OpenCL software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )

# Compilation flags
CXXFLAGS  := -O3 -Wall -Wextra -g -std=c++11 -fopenmp
NVCCFLAGS := -O3 -g -std=c++11 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fopenmp
NVCC_NONSUPPORTED_FLAGS := -Wl,--no-as-needed

# Compiler
CXX  := g++
NVCC := nvcc

OBJ_DIR := obj

# Target
TARGET := host
TARGET_DIR := bin

# Directories
INC_DIRS := ../common/inc ./host/include
LIB_DIRS := 

# Files
INCS := $(wildcard ./host/include/*.h)
SRCS := $(wildcard ./host/src/*.cc ../common/src/AOCLUtils/*.cc)
# OBJS := $(addprefix $(OBJ_DIR)/, $(notdir $(SRCS:.cc=.o)))
OBJS := $(SRCS:.cc=.o)
LIBS := rt


# Commandline arguments
########################################################################
NAME      := cg
# NUMDATA   := 1048576
NUMDATA   := 1024
NUMSTREAM := 4
NUMTRY    := 10


# Settings of the OpenCL kernel code compilation
########################################################################
CL_KERNEL              := ./fpga/$(NAME).cl
OFFLINE_COMPILER       := aoc
OFFLINE_COMPILER_FLAGS := -report -save-temps -dot -Werror -g -v 
TARGETED_FPGA_BOARD    := a10pl4_dd4gb_gx115_m512
OUTPUT_AOCX            := ./$(TARGET_DIR)/$(NAME).aocx


# Build host executable
########################################################################
# Make it all!
all: $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET): Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) $(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L) -c
	$(NVCC) $(NVCCFLAGS) $(foreach D,$(INC_DIRS),-I$D) host/src/main.cu -c
	$(NVCC) $(NVCCFLAGS) $(filter-out $(NVCC_NONSUPPORTED_FLAGS),$(AOCL_LINK_CONFIG)) -o $@ *.o

# # Host executable target.
# $(TARGET_DIR)/$(TARGET): $(OBJS)
# 	$(CXX) $(CXXFLAGS) -o $@ $^

# $(OBJ_DIR)/%.o: %.cc
# 	@if [ ! -d $(OBJ_DIR) ]; then echo "mkdir -p $(OBJ_DIR)"; mkdir -p $(OBJ_DIR); fi
# 	$(CXX) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L) -o $@ -c $<

$(TARGET_DIR):
	mkdir $(TARGET_DIR)


# Run or Debug
########################################################################
run:
	srun -p comq -w ppx00 $(TARGET_DIR)/$(TARGET) $(NAME) $(NUMDATA) $(NUMSTREAM) $(NUMTRY)

emu:
	srun -p comq -w ppx00 env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 $(TARGET_DIR)/$(TARGET) $(NAME) $(NUMDATA) $(NUMSTREAM) $(NUMTRY)

memcheck:
	srun -p comq -w ppx00 env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 valgrind -v --tool=memcheck --error-limit=no --leak-check=full --show-reachable=no --log-file=valgrind.log $(TARGET_DIR)/$(TARGET) $(NAME) $(NUMDATA) $(NUMSTREAM) $(NUMTRY)

debug:
	srun -p comq -w ppx00 env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 gdb --args $(TARGET_DIR)/$(TARGET) $(NAME) $(NUMDATA) $(NUMSTREAM) $(NUMTRY)


# Create FPGA configuration image data
########################################################################
gen:
	srun -p syn2 -w ppxsyn02 $(OFFLINE_COMPILER) -board=$(TARGETED_FPGA_BOARD) $(OFFLINE_COMPILER_FLAGS) $(CL_KERNEL) -o $(OUTPUT_AOCX)

# for emulation
egen:
	$(OFFLINE_COMPILER) -march=emulator $(OFFLINE_COMPILER_FLAGS) $(CL_KERNEL) -o $(OUTPUT_AOCX)


# Cleanup
########################################################################
# Standard make targets
clean :
	rm -f $(TARGET_DIR)/$(TARGET) valgrind.log *.o

.PHONY : all clean
