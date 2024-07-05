#include <iostream>

#include "dsaX.h"
#include "dsaX_params.h"
#include "dsaX_cuda_headers.h"
#include "dsaX_cuda_handles.h"
//#include "dsaX_cuda_kernels.h" // For debug

using namespace std;

__global__ void deviceInspectHalf(half *input, int stage) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  printf("CUBLAS[%d]: device inspect half [%d] =  %f\n", stage, x, __half2float(input[x])); 
}

void init_cublas_local() {
  if (!cublas_init) {
    //cublasError_t error = cudaStreamCreate(streams);
    cublasStatus_t error = cublasCreate(&cublasH);
    //cublasSetStream(handle, stream);
    //cublasStatus_t error = cublasCreate(&handle);
    if (error != CUBLAS_STATUS_SUCCESS)
      cout << "cublasCreate failed with error " << error << endl;
    else
      cout << "cublasCreated successfully." << endl;
    cublas_init = true;
  }
}

void destroy_cublas_local() {
  if(cublas_init)
    cublasDestroy(cublasH);
  cublas_init = false;
}

void initBLASCuda() {
  init_cublas_local();
}

using namespace std;

void dsaXHgemmStridedBatchedCuda(void *real_a, void *imag_a, void *real_b, void *imag_b, void *real_c, void *imag_c, dsaXBLASParam blas_param, int stream) {
#ifdef DSA_XENGINE_TARGET_CUDA
  
  // not sure if essential
  //cudaDeviceSynchronize();

  cublasSetStream(cublasH, get_stream(stream));

  bool verbose = false;
  
  // Set up for gemm
  //----------------
  // Transfer params
  const int m = blas_param.m;
  const int n = blas_param.n;
  const int k = blas_param.k;
  const double alpha = blas_param.alpha.real();
  const int lda = blas_param.lda;
  const int ldb = blas_param.ldb;
  const half beta0 = blas_param.beta.real();
  const half beta1 = 1.0;
  const int ldc = blas_param.ldc;
  const long long int a_offset = blas_param.a_offset;
  const long long int b_offset = blas_param.b_offset;
  const long long int c_offset = blas_param.c_offset;
  const long long int strideA = blas_param.a_stride;
  const long long int strideB = blas_param.b_stride;
  const long long int strideC = blas_param.c_stride;
  const int batchCount = blas_param.batch_count;

  // NOTE: cublasHgemm is a real valued kernel. As a result,
  // matrix conjugates must be handled by passing negative
  // alpha values on the appropriate imaginary planar
  // arrays. We discern these negative values while parsing
  // transpose, adjoint and conjugation values.
  cublasOperation_t transa;
  cublasOperation_t transb;
  int A_imag_alpha_sign = 1.0;
  switch (blas_param.trans_a) {
  case DSA_BLAS_OP_N:
    transa = CUBLAS_OP_N;
    break;
  case DSA_BLAS_OP_T:
    transa = CUBLAS_OP_T;
    break;
  case DSA_BLAS_OP_A:
    transa = CUBLAS_OP_N; 	
    // A array requests adjoint, hence we
    // must apply supply a factor of -1 to alpha
    // when dealing with the imaginary component
    // of A.
    A_imag_alpha_sign *= -1;
    break;
  case DSA_BLAS_OP_C:
    transa = CUBLAS_OP_T; 
    // A array requests conjugation, hence we
    // must apply supply a factor of -1 to alpha
    // when dealing with the imaginary component
    // of A.
    A_imag_alpha_sign *= -1;
    break;
  default:
    std::cout << "Unknown cublas transpose" << std::endl;
  }
  
  int B_imag_alpha_sign = alpha;
  switch (blas_param.trans_b) {
  case DSA_BLAS_OP_N:
    transb = CUBLAS_OP_N;
    break;
  case DSA_BLAS_OP_T:
    transb = CUBLAS_OP_T;
    break;
  case DSA_BLAS_OP_A:
    transb = CUBLAS_OP_N; 	
    // B array requests adjoint, hence we
    // must apply supply a factor of -1 to alpha
    // when dealing with the imaginary component
    // of B.
    B_imag_alpha_sign *= -1;
    break;
  case DSA_BLAS_OP_C:
    transb = CUBLAS_OP_T; 
    // A array requests conjugation, hence we
    // must apply supply a factor of -1 to alpha
    // when dealing with the imaginary component
    // of B.
    B_imag_alpha_sign *= -1;
    break;
  default:
    std::cout << "Unknown dsaBLAS transpose" << std::endl;
  }

  // Run strided batched gemm for datatype 
  // (a + ib)(c + id) = (ac - bd) + i(bc + ad)
  // on matrices C = alpha * op(A) * op(B) + beta * C
  // where op(M) is defined by the transposition variable
  // cublasOperation_t transM

  //deviceInspectHalf<<<1, 8>>>((half *)real_a);
  
  // Accumulate results into C matrix
  // ac
  half alpha_ac = alpha;
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &(alpha_ac),
			    (half *)real_a + a_offset, lda, strideA,
			    (half *)real_b + b_offset, ldb, strideB, &beta0,
			    (half *)real_c + c_offset, ldc, strideC,
			    batchCount);

  if(verbose) deviceInspectHalf<<<1, 8>>>((half *)real_c, 0);
  
  // -bd (minus sign from i*i)
  half alpha_bd = alpha * (-1.0 * A_imag_alpha_sign * B_imag_alpha_sign);
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &(alpha_bd),
			    (half*)imag_a + a_offset, lda, strideA,
			    (half*)imag_b + b_offset, ldb, strideB, &beta1,
			    (half*)real_c + c_offset, ldc, strideC,
			    batchCount);
  
  if(verbose) deviceInspectHalf<<<1, 8>>>((half *)real_c, 1);
  
  // bc
  half alpha_bc = alpha * A_imag_alpha_sign;
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &(alpha_bc),
			    (half*)imag_a + a_offset, lda, strideA,
			    (half*)real_b + b_offset, ldb, strideB, &beta0,
			    (half*)imag_c + c_offset, ldc, strideC,
			    batchCount);
  
  if(verbose) deviceInspectHalf<<<1, 8>>>((half *)imag_c, 2);
  
  // ad
  half alpha_ad = alpha * B_imag_alpha_sign;
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &(alpha_ad),
			    (half*)real_a + a_offset, lda, strideA,
			    (half*)imag_b + b_offset, ldb, strideB, &beta1,
			    (half*)imag_c + c_offset, ldc, strideC,
			    batchCount);

  if(verbose) deviceInspectHalf<<<1, 8>>>((half *)imag_c, 3);
  
  // shown to be essential (only with streams, fix me) 
  //cudaDeviceSynchronize();
#else
  std::cout "dsaX not built with CUDA target." << std::endl;
  exit(0);
#endif
}
