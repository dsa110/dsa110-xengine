#include <iostream>
#include <vector>
#include <cstring>
#include <string>

#include "dsaX_params.h"
#include "dsaX_cuda_interface.h"
#include "dsaX_utils.h"
#include "dsaX_ftd.h"

using namespace std;


void dsaXInit(int dev){
#if DSA_XENGINE_TARGET_CUDA
  dsaXInitCuda(dev);
#endif

  std::cout << " --- Starting dsaX with configuration (defined in dsaX_def.h) --- " << endl;
  std::cout << "NPACKETS_PER_BLOCK = " << NPACKETS_PER_BLOCK << std::endl;
  std::cout << "NCHAN = " << NCHAN << std::endl;
  std::cout << "NCHAN_PER_PACKET = " << NCHAN_PER_PACKET << std::endl;
  std::cout << "NPOL = " << NPOL << std::endl;
  std::cout << "NARM = " << 3 << std::endl;
  std::cout << " --- End dsaX configuration --- " << endl;
  //DMH: Add more (ask Vikram)
}

void dsaXEnd() {
  // output metrics
}

void inspectPackedData(char input, int i, bool non_zeros) {
  float re = (float)((char)((   (unsigned char)(input) & (unsigned char)(15)  ) << 4) >> 4);
  float im = (float)((char)((   (unsigned char)(input) & (unsigned char)(240))) >> 4);

  if(non_zeros) {
    if(re != 0 || im != 0) 
      std::cout << "val["<<i<<"] = ("<<re<<","<<im<<")" << std::endl;
  } else {
    std::cout << "val["<<i<<"] = ("<<re<<","<<im<<")" << std::endl;
  }
}

void dsaXCorrelator(void *output_data, void *input_data, dsaXCorrParam *param) {  

  dmem_corr d;
#if DSA_XENGINE_TARGET_CUDA  
  initializeCorrCudaMemory(&d);
  d.h_input = (char *)malloc(sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  memcpy(d.h_input, (char*)input_data, sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  dcorrelator(&d);
  dsaXmemcpy(output_data, d.d_output, NBASE*NCHAN_PER_PACKET*2*2*4, dsaXMemcpyDeviceToHost);
  deallocateCorrCudaMemory(&d);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void reorderCorrInput(dmem_corr *d) {
#if DSA_XENGINE_TARGET_CUDA
  reorderCorrInputCuda(d);
#else
  std::cout << "dsaX error: not implemented" << std::endl;
#endif
}

void reorderCorrOutput(dmem_corr *d) {
#if DSA_XENGINE_TARGET_CUDA  
  reorderCorrOutputCuda(d);
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
