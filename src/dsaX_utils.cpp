#include "dsaX_utils.h"
#ifdef DSA_XENGINE_TARGET_CUDA
#include "dsaX_cuda_headers.h"
#endif

void dsaXmemset(void *array, int ch, size_t n){
#ifdef DSA_XENGINE_TARGET_CUDA
  cudaMemset(array, ch, n);
#else
  emset(array, ch, n);
#endif
}

void dsaXmemcpyHostToDevice(void *array_device, void *array_host, size_t n){
#ifdef DSA_XENGINE_TARGET_CUDA
  // Perform host to device memcopy on data
  cudaMemcpy(array_device, array_host, n, cudaMemcpyHostToDevice);
#else  
  memcpy(array_device, array_host, n);
#endif
}

void dsaXmemcpyDeviceToHost(void *array_host, void *array_device, size_t n){
#ifdef DSA_XENGINE_TARGET_CUDA
  // Perform host to device memcopy on data
  cudaMemcpy(array_host, array_device, n, cudaMemcpyDeviceToHost);
#else
  memcpy(array_host, array_device, n);
#endif
}

void dsaXmemcpyDeviceToDevice(void *array_copy_to, void *array_copy_from, size_t n){
#ifdef DSA_XENGINE_TARGET_CUDA
  // Perform device to device memcopy on data
  cudaMemcpy(array_copy_to, array_copy_from, n, cudaMemcpyDeviceToDevice);
#else
  memcpy(array_copy_to, array_copy_from, n);
#endif
}
