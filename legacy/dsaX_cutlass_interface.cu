/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "dsaX_cutlass_interface.h"

DSA_FTD_ComplexGEMM_CUTLASS::DSA_FTD_ComplexGEMM_CUTLASS(Options const &options): 
  problem_size(options.problem_size), batch_count(options.batch_count) {

  // Allocate device memory for batched planar complex GEMM  
  tensor_A.reset(int64_t(problem_size.m()) * problem_size.k() * batch_count * 2);
  tensor_B.reset(int64_t(problem_size.k()) * problem_size.n() * batch_count * 2);
  tensor_C.reset(int64_t(problem_size.m()) * problem_size.n() * batch_count * 2);
  tensor_D.reset(int64_t(problem_size.m()) * problem_size.n() * batch_count * 2);
  tensor_D_ref.reset(int64_t(problem_size.m()) * problem_size.n() * batch_count * 2);
  
  ptr_A_real.reset(batch_count);
  ptr_A_imag.reset(batch_count);
  ptr_B_real.reset(batch_count);
  ptr_B_imag.reset(batch_count);
  ptr_C_real.reset(batch_count);
  ptr_C_imag.reset(batch_count);
  ptr_D_real.reset(batch_count);
  ptr_D_imag.reset(batch_count);      
}

// DMH: Replace this with data from DSA-FTD
void DSA_FTD_ComplexGEMM_CUTLASS::initialize() {

  if(testing) {
    uint64_t seed = 1234;
    
    // Use small integers to simplify correctness checking
    int scope_max = 6;
    int scope_min = -6;
    
    BlockFillRandomUniform(tensor_A.get(), tensor_A.size(), seed, Element(scope_max), Element(scope_min), 0);
    BlockFillRandomUniform(tensor_B.get(), tensor_B.size(), seed * 2019, Element(scope_max), Element(scope_min), 0);
    BlockFillRandomUniform(tensor_C.get(), tensor_C.size(), seed * 2020, Element(scope_max), Element(scope_min), 0);
  } else {
    // DMH: construct DSA-FTD interface data transfer interface
  }

  ptr_A = tensor_A.get();
  ptr_B = tensor_B.get();
  ptr_C = tensor_C.get();
  ptr_D = tensor_D.get();
  
  batch_stride_A = int64_t(problem_size.m()) * problem_size.k() * 2;
  batch_stride_B = int64_t(problem_size.k()) * problem_size.n() * 2;
  batch_stride_C = int64_t(problem_size.m()) * problem_size.n() * 2;
  batch_stride_D = int64_t(problem_size.m()) * problem_size.n() * 2;
  
  lda = LayoutA::packed({problem_size.m(), problem_size.k()}).stride(0);
  ldb = LayoutB::packed({problem_size.k(), problem_size.n()}).stride(0);
  ldc = LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);
  ldd = LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);
  
  imag_stride_A = int64_t(problem_size.m()) * problem_size.k();
  imag_stride_B = int64_t(problem_size.k()) * problem_size.n();
  imag_stride_C = int64_t(problem_size.m()) * problem_size.n();
  imag_stride_D = int64_t(problem_size.m()) * problem_size.n();

}

Result DSA_FTD_ComplexGEMM_CUTLASS::run(Options const &options) {
  
  Result result;
  
  initialize();  

  // Configure pointers in global memory
  struct {
    Element *base;
    void **ptr_real;
    void **ptr_imag;
    int64_t batch_stride;
    int64_t imag_stride;
  } tensors[] = {{ tensor_A.get(), ptr_A_real.get(), ptr_A_imag.get(), batch_stride_A, imag_stride_A},
		 { tensor_B.get(), ptr_B_real.get(), ptr_B_imag.get(), batch_stride_B, imag_stride_B},
		 { tensor_C.get(), ptr_C_real.get(), ptr_C_imag.get(), batch_stride_C, imag_stride_C},
		 { tensor_D.get(), ptr_D_real.get(), ptr_D_imag.get(), batch_stride_D, imag_stride_D}};
  
  for (auto const &tensor : tensors) {
    for (int idx = 0; idx < batch_count; ++idx) {
      
      cudaError_t error;
      void *ptr_real = tensor.base + idx * tensor.batch_stride;
      void *ptr_imag = tensor.base + idx * tensor.batch_stride + tensor.imag_stride;      
      
      error = cudaMemcpy(tensor.ptr_real + idx, &ptr_real, sizeof(void *), cudaMemcpyHostToDevice);
      if (error != cudaSuccess) throw std::runtime_error("Failed to copy pointer to device memory");
      
      error = cudaMemcpy(tensor.ptr_imag + idx, &ptr_imag, sizeof(void *), cudaMemcpyHostToDevice);
      if (error != cudaSuccess) throw std::runtime_error("Failed to copy pointer to device memory");
      
    }
  }

  
  cudaEvent_t events[2];  
  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
  }
  
  // Record an event at the start of a series of GEMM operations
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }

  // Run profiling loop
  //-------------------
  // Execute the planar complex array GEMM kernel via the CUTLASS Library's
  // dispatch routines.
  //
  // Note, for planar complex array GEMM kernels, all numeric type arguments 
  // specify the data type of the base real types. These are understood to
  // apply to planar complex representations of matrices in memory and to complex<T>
  // structures for scalars.
  //
  // See tools/library/include/cutlass/library/handle.h for more details.
  //
  for (int iter = 0; iter < options.iterations; ++iter) {
    
    result.status = handle.gemm_planar_complex_array(
	problem_size.m(),                                 // expected GEMM M dimension
	problem_size.n(),                                 // expected GEMM N dimension
	problem_size.k(),                                 // expected GEMM K dimension
	batch_count,                                      // Number of batched elements

        nullptr,
        nullptr,
        nullptr,

        cutlass::library::NumericTypeID::kF32,            // Base data type of complex-valued accumulation
        cutlass::library::NumericTypeID::kF32,            // Base data type of complex-valued alpha/beta scalars

        &options.alpha,                                   // Pointer to alpha scalar, of type complex<T>

        cutlass::library::NumericTypeID::kF16,            // Base data type of complex-valued A matrix
        cutlass::library::LayoutTypeID::kColumnMajor,     // Layout of A matrix
        cutlass::library::ComplexTransform::kConjugate,   // Complex transformation on A matrix operand

        ptr_A_real.get(),                                 // Pointer to array of pointers to real part of A matrix
        ptr_A_imag.get(),                                 // Pointer to array of pointers to imaginary part of A matrix

        lda,                                              // Leading dimension of real part of A matrix
        lda,                                              // Leading dimension of imaginary part of A matrix

        cutlass::library::NumericTypeID::kF16,            // Base data type of complex-valued B matrix
        cutlass::library::LayoutTypeID::kColumnMajor,     // Layout of B matrix
        cutlass::library::ComplexTransform::kNone,        // Complex transformation on B matrix operand

        ptr_B_real.get(),                                 // Pointer to array of pointers to real part of B matrix
        ptr_B_imag.get(),                                 // Pointer to array of pointers to imaginary part of B matrix

        ldb,                                              // Leading dimension of real part of B matrix
        ldb,                                              // Leading dimension of imaginary part of B matrix

        &options.beta,                                    // Pointer to beta scalar, of type complex<T>

        cutlass::library::NumericTypeID::kF16,            // Base data type of complex valued C and D matrices

        ptr_C_real.get(),                                 // Pointer to array of pointers to real part of C matrix
        ptr_C_imag.get(),                                 // Pointer to array of pointers to imaginary part of C matrix

        ldc,                                              // Leading dimension of real part of C matrix
        ldc,                                              // Leading dimension of imaginary part of C matrix

        ptr_D_real.get(),                                 // Pointer to array of pointers to real part of D matrix
        ptr_D_imag.get(),                                 // Pointer to array of pointers to imaginary part of D matrix

        ldd,                                              // Leading dimension of real part of D matrix
        ldd                                               // Leading dimension of imaginary part of D matrix
						     );
    
    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "CUTLASS internal error - configuration not supported" << std::endl;
      return result;
    }
  }
  
  // Record an event when the GEMM operations have been launched.
  result.error = cudaEventRecord(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }
  
  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }
  
  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }
  
  // Compute average runtime and GFLOPs.
  result.runtime_ms = double(runtime_ms) / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);
  
  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  
  if (handle.get_last_operation()) {
    std::cout << "Recently executed '" << handle.get_last_operation()->description().name << "'" << std::endl;
  }

  // Compute reference in device code
  if (options.reference_check) {
    
    result.passed = true;
    
    for (int64_t idx = 0; result.passed && idx < int64_t(batch_count); ++idx) {
      // Define the GEMM through templates
      GemmPlanarComplex<Element, LayoutA, Element, LayoutB, Element, LayoutC, ElementAccumulator>
	(problem_size, options.alpha,
	 {tensor_A.get() + idx * batch_stride_A, lda, imag_stride_A},
	 cutlass::ComplexTransform::kConjugate,
	 {tensor_B.get() + idx * batch_stride_B, ldb, imag_stride_B},
	 cutlass::ComplexTransform::kNone,
	 options.beta,
	 {tensor_C.get() + idx * batch_stride_C, ldc, imag_stride_C},
	 {tensor_D_ref.get() + idx * batch_stride_D, ldd, imag_stride_D}
	 );
      
      Element epsilon = 0.1_hf;
      Element nonzero_floor = 0.1_hf;
      
      result.passed = BlockCompareRelativelyEqual
	(
	 tensor_D.get() + idx * batch_stride_D,
	 tensor_D_ref.get() + idx * batch_stride_D,
	 batch_stride_D,
	 epsilon,
	 nonzero_floor
	 );
    }
    
    if (result.passed) std::cout << "Reference check passed." << std::endl;
    else std::cerr << "Error - reference check failed." << std::endl;
  }
  
  std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
  std::cout << " GFLOPs: " << result.gflops << std::endl;
  
  return result;
}

 int main(int argc, char const **args) {
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
  
  Options options;  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // Compute GEMM
  DSA_FTD_ComplexGEMM_CUTLASS gemm(options);
  gemm.testing = true;
  Result result = gemm.run(options);
  
  return result.passed ? 0 : -1;
}
