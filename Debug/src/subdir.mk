################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/MemoryMonitor.cpp \
../src/costGradient.cpp \
../src/hardware.cpp \
../src/resultPredict.cpp 

CU_SRCS += \
../src/InputInit.cu \
../src/cuMath.cu \
../src/cuMatrix.cu 

CC_SRCS += \
../src/cost_gradient.cc \
../src/gradient_checking.cc \
../src/helper.cc \
../src/matrix_maths.cc \
../src/read_config.cc \
../src/read_data.cc \
../src/result_predict.cc \
../src/sample.cc \
../src/train_network.cc \
../src/weight_init.cc \
../src/weights_IO.cc 

CU_DEPS += \
./src/InputInit.d \
./src/cuMath.d \
./src/cuMatrix.d 

OBJS += \
./src/InputInit.o \
./src/MemoryMonitor.o \
./src/costGradient.o \
./src/cost_gradient.o \
./src/cuMath.o \
./src/cuMatrix.o \
./src/gradient_checking.o \
./src/hardware.o \
./src/helper.o \
./src/matrix_maths.o \
./src/read_config.o \
./src/read_data.o \
./src/resultPredict.o \
./src/result_predict.o \
./src/sample.o \
./src/train_network.o \
./src/weight_init.o \
./src/weights_IO.o 

CC_DEPS += \
./src/cost_gradient.d \
./src/gradient_checking.d \
./src/helper.d \
./src/matrix_maths.d \
./src/read_config.d \
./src/read_data.d \
./src/result_predict.d \
./src/sample.d \
./src/train_network.d \
./src/weight_init.d \
./src/weights_IO.d 

CPP_DEPS += \
./src/MemoryMonitor.d \
./src/costGradient.d \
./src/hardware.d \
./src/resultPredict.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/cuda/samples/common/inc/ -I/usr/local/include/ -G -g -O0 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/cuda/samples/common/inc/ -I/usr/local/include/ -G -g -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/cuda/samples/common/inc/ -I/usr/local/include/ -G -g -O0 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/cuda/samples/common/inc/ -I/usr/local/include/ -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/cuda/samples/common/inc/ -I/usr/local/include/ -G -g -O0 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/cuda/samples/common/inc/ -I/usr/local/include/ -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


