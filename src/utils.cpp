#include <iostream>

#include "utils.h"
#include "enums.h"
#include "params.h"
#include "cuda_interface.h"

using namespace std;

void dsaXmemset(void *array, int ch, size_t n){
#ifdef DSA_XENGINE_TARGET_CUDA
  dsaXmemsetCuda(array, ch, n);
#else
  memset(array, ch, n);
#endif
}

void dsaXmemcpy(void *array_out, void *array_in, size_t n, dsaXMemcpyKind kind, int stream){

#ifdef DSA_XENGINE_TARGET_CUDA
  // Perform host to device memcopy on data
  dsaXmemcpyCuda(array_out, array_in, n, kind, stream);
#else  
  memcpy(array_out, array_in, n);
#endif
}

void dsaXDeviceSynchronize() {
#ifdef DSA_XENGINE_TARGET_CUDA
  // Synchronise the device
  dsaXDeviceSynchronizeCuda();
#else  
  // NO OP
#endif
}

void initDsaXCorrDeviceMemory(corr_handle *d, unsigned int n_streams) {

#ifdef DSA_XENGINE_TARGET_CUDA
  d->dev_malloc_timer.start();
  initializeCorrCudaMemory(d, n_streams);
  d->dev_malloc_timer.stop();
#else  
  cout << "dsaX Error: Not implemented." << endl;
  exit(0);
#endif  
}

void destroyDsaXCorrDeviceMemory(corr_handle *d) {

#ifdef DSA_XENGINE_TARGET_CUDA
  d->dev_malloc_timer.start();
  deallocateCorrCudaMemory(d);
  d->dev_malloc_timer.stop();
#else
  cout << "dsaX Error: Not implemented." << endl;
  exit(0);
#endif  
}
