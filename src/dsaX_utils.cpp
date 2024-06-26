#include "dsaX_utils.h"
#include "dsaX_enums.h"
#include "dsaX_cuda_interface.h"

void dsaXmemset(void *array, int ch, size_t n){
#ifdef DSA_XENGINE_TARGET_CUDA
  dsaXmemsetCuda(array, ch, n);
#else
  memset(array, ch, n);
#endif
}

void dsaXmemcpy(void *array_out, void *array_in, size_t n, dsaXMemcpyKind kind){
#ifdef DSA_XENGINE_TARGET_CUDA
  // Perform host to device memcopy on data
  dsaXmemcpyCuda(array_out, array_in, n, kind);
#else  
  memcpy(array_out, array_in, n);
#endif
}

void dsaXDeviceSynchronize() {
#ifdef DSA_XENGINE_TARGET_CUDA
  // Perform host to device memcopy on data
  dsaXDeviceSynchronizeCuda();
#else  
  // NO OP
#endif
}
