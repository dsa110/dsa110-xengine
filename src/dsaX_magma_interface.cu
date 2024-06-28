#include <iostream>
#include "dsaX.h"
#include "dsaX_cuda_headers.h"
#include "dsaX_magma_headers.h"

using namespace std;

void dsaXHgemmStridedBatchedMagma(void *real_a, void *imag_a, void *real_b, void *imag_b, void *real_c, void *imag_c, dsaXBLASParam blas_param) {
#if defined (DSA_XENGINE_TARGET_CUDA)
#if defined (DSA_XENGINE_ENABLE_MAGMA)

  // TO DO
  
#else
  std::cout << "dsaX not built with MAGMA. Rebuild with CMake param DSA_XENGINE_ENABLE_MAGMA=ON" << std::endl;
  exit(0);
#endif
#else
  std::cout << "dsaX not built with CUDA target. Rebuild with CMake param DSA_XENGINE_TARGET_TYPE=CUDA" << std::endl;
  exit(0);
#endif
}
