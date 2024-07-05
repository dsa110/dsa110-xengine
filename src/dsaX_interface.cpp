#include <iostream>
#include <vector>
#include <cstring>
#include <string>

#include "dsaX_params.h"
#include "dsaX_cuda_interface.h"
#include "dsaX_utils.h"
#include "dsaX_ftd.h"

using namespace std;

using ms = std::chrono::microseconds;
using hrc = std::chrono::high_resolution_clock;  

timer::Timer<ms, hrc> app_timer;
timer::Timer<ms, hrc> init_timer;

void dsaXInit(int dev){
  app_timer.start();
#if DSA_XENGINE_TARGET_CUDA
  init_timer.start();
  dsaXInitCuda(dev);
  initBLAS();
  init_timer.stop();
#endif
  cout << " --- Starting dsaX with configuration (defined in dsaX_def.h) --- " << endl;
  cout << "NPACKETS_PER_BLOCK = " << NPACKETS_PER_BLOCK << endl;
  cout << "NCHAN = " << NCHAN << endl;
  cout << "NCHAN_PER_PACKET = " << NCHAN_PER_PACKET << endl;
  cout << "NPOL = " << NPOL << endl;
  cout << "NARM = " << 2 << endl;
#if DSA_XENGINE_TARGET_CUDA
  cout << "CUDA is ENABLED " << endl;
#else
  cout << "CUDA is DISABLED " << endl;
#endif
  cout << " --- End dsaX configuration --- " << endl;
  //DMH: Add more (ask Vikram)
}

void dsaXEnd() {
  app_timer.stop();
  // output metrics
  cout << "dsaX lifetime = " << (1.0*app_timer.elapsed().count())/(1e6) << endl;
  cout << "dsaX init = " << (1.0*init_timer.elapsed().count())/(1e6) << endl;
}

void *dsaXHostRegister(size_t size) {
#if DSA_XENGINE_TARGET_CUDA  
  return dsaXHostRegisterCuda(size);
#endif
}

void inspectPackedData(char input, int i, bool non_zeros) {
  float re = (float)((char)((   (unsigned char)(input) & (unsigned char)(15)  ) << 4) >> 4);
  float im = (float)((char)((   (unsigned char)(input) & (unsigned char)(240))) >> 4);
  
  if(non_zeros) {
    if(re != 0 || im != 0) 
      cout << "val["<<i<<"] = ("<<re<<","<<im<<")" << endl;
  } else {
    cout << "val["<<i<<"] = ("<<re<<","<<im<<")" << endl;
  }
}

void promoteComplexCharToPlanarHalf(corr_handle *d, unsigned int stream) {
#if DSA_XENGINE_TARGET_CUDA
  promoteComplexCharToPlanarHalfCuda(d, stream);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}

void reorderCorrInput(corr_handle *d, int stream) {
#if DSA_XENGINE_TARGET_CUDA
  reorderCorrInputCuda(d, stream);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}

void initBLAS() {
#if DSA_XENGINE_TARGET_CUDA
  // DMH: Fix me for orther libs
  initBLASCuda();
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}

void initStreams(unsigned int n_streams) {
#if DSA_XENGINE_TARGET_CUDA
  initStreamsCuda(n_streams);
#else
  // NO OP
#endif
}

void destroyStreams() {
#if DSA_XENGINE_TARGET_CUDA
  destroyStreamsCuda();
#else
  // NO OP
#endif
}

void computeIndices(corr_handle *d) {
#if DSA_XENGINE_TARGET_CUDA
  computeIndicesCuda(d);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}


void reorderCorrOutput(corr_handle *d, int stream) {
#if DSA_XENGINE_TARGET_CUDA  
  reorderCorrOutputCuda(d, stream);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}

void transposeInputBeamformer(double *input, double *output, vector<int> &dimBlock, vector<int> &dimGrid) {
#if DSA_XENGINE_TARGET_CUDA
  transposeInputBeamformerCuda(input, output, dimBlock, dimGrid);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}

void transposeScaleBeamformer(void *real, void *imag, unsigned char *output, vector<int> &dimBlock, vector<int> &dimGrid) {
#if DSA_XENGINE_TARGET_CUDA
  transposeScaleBeamformerCuda(real, imag, output, dimBlock, dimGrid);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}

void fluffInputBeamformer(char *input, void *array_real, void *array_imag, int blocks, int tpb) {
#if DSA_XENGINE_TARGET_CUDA
  fluffInputBeamformerCuda(input, array_real, array_imag, blocks, tpb);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}

void sumBeam(unsigned char *input, float *output, int blocks, int tpb) {
#if DSA_XENGINE_TARGET_CUDA
  sumBeamCuda(input, output, blocks, tpb);
#else
  cout << "dsaX error: not implemented" << endl;
#endif
}
