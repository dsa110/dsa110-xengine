#include <iostream>
#include "dsaX_cublas_interface.h"

using namespace std;

void dsaXHgemmStridedBatchedCuda(half *real_in, half *imag_in, half *real_out, half *imag_out, dsaXBLASParam blas_param) {
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
  const half malpha = -1.0 * blas_param.alpha.real();
  const int lda = blas_param.lda;
  const int ldb = blas_param.ldb;
  const half beta0 = blas_param.beta.real();
  const half beta1 = 1.0;
  const int ldc = blas_param.ldc;
  const long long int strideA = blas_param.a_stride;
  const long long int strideB = blas_param.b_stride;
  const long long int strideC = blas_param.c_stride;
  const int batchCount = blas_param.batch_count;
  
  // run strided batched gemm for datatype (a + ib)(c + id)
  // ac
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,real_in,lda,strideA,
			    real_in,ldb,strideB,&beta0,
			    real_out,ldc,strideC,
			    batchCount);
  // bd
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,imag_in,lda,strideA,
			    imag_in,ldb,strideB,&beta1,
			    real_out,ldc,strideC,
			    batchCount);
  // -bc
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &malpha,imag_in,lda,strideA,
			    real_in,ldb,strideB,&beta0,
			    imag_out,ldc,strideC,
			    batchCount);
  // ad
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,real_in,lda,strideA,
			    imag_in,ldb,strideB,&beta1,
			    imag_out,ldc,strideC,
			    batchCount);

  // shown to be essential
  cudaDeviceSynchronize();

  // destroy stream
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);  
#else
  std::cout "Not implemented" << std::endl;
  exit(0);
#endif
}
