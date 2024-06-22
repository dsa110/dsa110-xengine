#include <dsaX.h>
#include "dsaX_cublas_interface.h"

void dsaXHgemmStridedBatched(void *real_in, void *imag_in, void *real_out, void *imag_out, dsaXBLASParam param) {
#ifdef DSA_XENGINE_TARGET_CUDA
  dsaXHgemmStridedBatchedCuda(real_in, imag_in, real_out, imag_out, param);
#else
  std::cout "Not implemented" << std::endl;
  exit(0);
#endif
}
