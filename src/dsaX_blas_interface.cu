#include <dsaX.h>
#include "dsaX_cublas_interface.h"

void dsaXHgemmStridedBatched(void *real_in, void *imag_in, void *real_out, void *imag_out, dsaXBLASParam param) {
#ifdef DSA_XENGINE_TARGET_CUDA
  dsaXHgemmStridedBatchedCuda((half*)real_in, (half*)imag_in, (half*)real_out, (half*)imag_out, param);
#else
  std::cout "Not implemented" << std::endl;
  exit(0);
#endif
}
