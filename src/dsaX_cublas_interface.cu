#include <iostream>
#include "dsaX.h"
#include "dsaX_cuda_headers.h"

using namespace std;

void dsaXHgemmStridedBatchedCuda(void *real_a, void *imag_a, void *real_b, void *imag_b, void *real_c, void *imag_c, dsaXBLASParam blas_param) {
#ifdef DSA_XENGINE_TARGET_CUDA
  
  // not sure if essential
  cudaDeviceSynchronize();
  
  // Set up for gemm
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  // Transfer params
  cublasOperation_t transa;
  cublasOperation_t transb;
  switch (blas_param.trans_a) {
  case DSA_BLAS_OP_N:
    transa = CUBLAS_OP_N; break;
  case DSA_BLAS_OP_T:
    transa = CUBLAS_OP_T; break;
  case DSA_BLAS_OP_C:
    transa = CUBLAS_OP_C; break;
  default:
    std::cout << "Unknown cublas transpose" << std::endl;
  }

  switch (blas_param.trans_b) {
  case DSA_BLAS_OP_N:
    transb = CUBLAS_OP_N; break;
  case DSA_BLAS_OP_T:
    transb = CUBLAS_OP_T; break;
  case DSA_BLAS_OP_C:
    transb = CUBLAS_OP_C; break;
  default:
    std::cout << "Unknown cublas transpose" << std::endl;
  }
  
  const int m = blas_param.m;
  const int n = blas_param.n;
  const int k = blas_param.k;
  const half alpha = blas_param.alpha.real();
  const half malpha = (-1.0 * blas_param.alpha.real());
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
  
  // Run strided batched gemm for datatype 
  // (a + ib)(c + id) = (ac - bd) + i(bc + ad)
  // on matrices alpha * op(A) * op(B) + beta * C
  // where op(M) is defined by the transposition variable
  // cublasOperation_t transM
  
  // Accumulate results into C matrix
  // ac
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &alpha,
			    (half *)real_a + a_offset, lda, strideA,
			    (half *)real_b + b_offset, ldb, strideB, &beta0,
			    (half *)real_c + c_offset, ldc, strideC,
			    batchCount);
  // -bd
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &malpha,
			    (half*)imag_a + a_offset, lda, strideA,
			    (half*)imag_b + b_offset, ldb, strideB, &beta1,
			    (half*)real_c + c_offset, ldc, strideC,
			    batchCount);
  // bc
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &alpha,
			    (half*)imag_a + a_offset, lda, strideA,
			    (half*)real_b + b_offset, ldb, strideB, &beta0,
			    (half*)imag_c + c_offset, ldc, strideC,
			    batchCount);
  // ad
  cublasHgemmStridedBatched(cublasH, transa, transb, m,n,k, &alpha,
			    (half*)real_a + a_offset, lda, strideA,
			    (half*)imag_b + b_offset, ldb, strideB, &beta1,
			    (half*)imag_c + c_offset, ldc, strideC,
			    batchCount);
  
  // shown to be essential
  cudaDeviceSynchronize();
  
  // destroy stream
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);  
#else
  std::cout "dsaX not built with CUDA target." << std::endl;
  exit(0);
#endif
}
