/*
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

int main() {

  cutlass::half_t x = 2.25_hf;

  std::cout << x << std::endl;

  return 0;
}
*/

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main() {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm75                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //
  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

  cutlass::half_t const *ptrA = A.device_data();
  cutlass::half_t const *ptrB = B.device_data();
  cutlass::half_t const *ptrC = C.device_data();
  cutlass::half_t       *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = C.device_ref().stride(0);
  //
  // Launch GEMM on the device
  //
 
  status = gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  } else {
    std::cout << "CUTLASS Success! " << std::endl;
  }
  
  return 0;
}
