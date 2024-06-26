#include <iostream>
#include <vector>
#include <cstring>

#include "dsaX_cuda_interface.h"
#include "dsaX_ftd.h"

using namespace std;

void dsaXCorrelator(void *output_data, void *input_data) {  
  dmem d;
  int bf = 0;
#if DSA_XENGINE_TARGET_CUDA
  initializeCudaMemory(&d, bf);
  d.h_input = (char *)malloc(sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  memcpy(d.h_input, (char*)input_data, sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  dcorrelator(&d);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void reorderInput(dmem *d) {
#if DSA_XENGINE_TARGET_CUDA
  reorderInputCuda(d);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void reorderOutput(dmem *d) {
#if DSA_XENGINE_TARGET_CUDA  
  reorderOutputCuda(d);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void transposeInputBeamformer(double *input, double *output, std::vector<int> &dimBlock, std::vector<int> &dimGrid) {
#if DSA_XENGINE_TARGET_CUDA
  transposeInputBeamformerCuda(input, output, dimBlock, dimGrid);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void transposeScaleBeamformer(void *real, void *imag, unsigned char *output, std::vector<int> &dimBlock, std::vector<int> &dimGrid) {
#if DSA_XENGINE_TARGET_CUDA
  transposeScaleBeamformerCuda(real, imag, output, dimBlock, dimGrid);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void fluffInputBeamformer(char *input, void *array_real, void *array_imag, int blocks, int tpb) {
#if DSA_XENGINE_TARGET_CUDA
  fluffInputBeamformerCuda(input, array_real, array_imag, blocks, tpb);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void sumBeam(unsigned char *input, float *output, int blocks, int tpb) {
#if DSA_XENGINE_TARGET_CUDA
  sumBeamCuda(input, output, blocks, tpb);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}
