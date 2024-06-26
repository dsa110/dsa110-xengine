#pragma once

#include <iostream>
#include <sstream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor_planar_complex.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/gemm_planar_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/library/handle.h"

using namespace cutlass;
using namespace gemm;
using namespace library;
using namespace layout;
using namespace reference;
using namespace device;

// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  Status status;
  cudaError_t error;
  bool passed;
  
  Result(double runtime_ms = 0, double gflops = 0, Status status = Status::kSuccess, cudaError_t error = cudaSuccess):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

// Command line options parsing (testing)
struct Options {

  bool help;
  GemmCoord problem_size;
  int batch_count;
  complex<float> alpha;
  complex<float> beta;
  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_size({1024, 1024, 1024}),
    batch_count(256),
    reference_check(false),
    iterations(2),
    alpha(1),
    beta(0) { }

  // Parses the command line
  void parse(int argc, char const **args) {
    
    CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }
    
    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("batch", batch_count);

    cmd.get_cmd_line_argument("alpha", alpha.real());
    cmd.get_cmd_line_argument("alpha_i", alpha.imag());
    cmd.get_cmd_line_argument("beta", beta.real());
    cmd.get_cmd_line_argument("beta_i", beta.imag());
    
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "dsaX_cutlass_interface\n\n"
	<< "  This example uses the CUTLASS Library to execute Planar Complex Array GEMM computations.\n\n"
	<< "Options:\n\n"
	<< "  --help                      If specified, displays this usage statement.\n\n"
	<< "  --m=<int>                   GEMM M dimension\n"
	<< "  --n=<int>                   GEMM N dimension\n"
	<< "  --k=<int>                   GEMM K dimension\n"
	<< "  --batch=<int>               Number of GEMM operations executed in one batch\n"
	<< "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
	<< "  --alpha_i=<f32>             Epilogue scalar alpha (imaginary part)\n"
	<< "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
	<< "  --beta_i=<f32>              Epilogue scalar beta (imaginary part)\n\n"
	<< "  --iterations=<int>          Number of profiling iterations to perform.\n";
    
    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {
    
    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count * 4;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/// Performance test environment for planar complex
class DSA_FTD_ComplexGEMM_CUTLASS {

  // Half-precision input and output
  using Element = half_t;
  
  // Configurations for layouts and internal computation
  using LayoutA = ColumnMajor;
  using LayoutB = ColumnMajor;
  using LayoutC = ColumnMajor;
  using ElementCompute = float;
  using ElementAccumulator = float;

  Handle handle;
  
  GemmCoord problem_size;
  int batch_count;
  DeviceAllocation<Element> tensor_A;
  DeviceAllocation<Element> tensor_B;
  DeviceAllocation<Element> tensor_C;
  DeviceAllocation<Element> tensor_D;
  DeviceAllocation<Element> tensor_D_ref;

  DeviceAllocation<void *> ptr_A_real;
  DeviceAllocation<void *> ptr_A_imag;
  DeviceAllocation<void *> ptr_B_real;
  DeviceAllocation<void *> ptr_B_imag;
  DeviceAllocation<void *> ptr_C_real;
  DeviceAllocation<void *> ptr_C_imag;
  DeviceAllocation<void *> ptr_D_real;
  DeviceAllocation<void *> ptr_D_imag;

  Element *ptr_A;
  Element *ptr_B;
  Element *ptr_C;
  Element *ptr_D;
  
  int64_t batch_stride_A;
  int64_t batch_stride_B;
  int64_t batch_stride_C;
  int64_t batch_stride_D;
  
  typename LayoutA::Stride::Index lda;
  typename LayoutB::Stride::Index ldb;
  typename LayoutC::Stride::Index ldc;
  typename LayoutC::Stride::Index ldd;
  
  int64_t imag_stride_A;
  int64_t imag_stride_B;
  int64_t imag_stride_C;
  int64_t imag_stride_D;
  
public:  
  // Constructors
  DSA_FTD_ComplexGEMM_CUTLASS(Options const &options);
  DSA_FTD_ComplexGEMM_CUTLASS();
  
  // Methods
  void initialize();  
  Result run(Options const &options);
  
  bool testing;  
};
  
