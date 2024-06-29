#include <iostream>

#include "dsaX.h"
#include "dsaX_cublas_interface.h"
#include "dsaX_magma_interface.h"

void dsaXHgemmStridedBatched(void *real_a, void *imag_a, void *real_b, void *imag_b, void *real_c, void *imag_c, dsaXBLASParam param) {
  switch (param.blas_lib) {
  case DSA_BLAS_LIB_CUBLAS:
    dsaXHgemmStridedBatchedCuda(real_a, imag_a, real_b, imag_b, real_c, imag_c, param);
    break;
  case DSA_BLAS_LIB_MAGMA:
    dsaXHgemmStridedBatchedMagma(real_a, imag_a, real_b, imag_b, real_c, imag_c, param);
    break;
  case DSA_BLAS_LIB_CUTLASS:
    //dsaXHgemmStridedBatchedCutlass(real_a, imag_a, real_b, imag_b, real_c, imag_c, param);
    break;
  case DSA_BLAS_LIB_OPENBLAS:
    //dsaXHgemmStridedBatchedOpenblas(real_a, imag_a, real_b, imag_b, real_c, imag_c, param);
    break;
  case DSA_BLAS_LIB_TCC:
    //dsaXHgemmStridedBatchedTcc(real_a, imag_a, real_b, imag_b, real_c, imag_c, param);
    break;
  default:
    std::cout << "dsaX Error: Unknown blas_lib " << param.blas_lib << " given." << std::endl;
    exit(0);
  }
}
