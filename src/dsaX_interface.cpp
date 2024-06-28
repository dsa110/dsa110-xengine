#include <iostream>
#include <vector>
#include <cstring>

#include "dsaX_cuda_interface.h"
#include "dsaX_utils.h"
#include "dsaX_ftd.h"

using namespace std;

void printDsaXBLASParam(const dsaXBLASParam param) {

  cout << "struct_size = " << param.struct_size << endl;
  cout << "blas_type = " << param.blas_type << endl;
  cout << "blas_lib = " << param.blas_lib << endl;
  cout << "data_order = " << param.data_order << endl;
  cout << "trans_a = " << param.trans_a << endl;
  cout << "trans_b = " << param.trans_b << endl;
  cout << "m = " << param.m << endl;
  cout << "n = " << param.n << endl;
  cout << "k = " << param.k << endl;
  cout << "lda = " << param.lda << endl;
  cout << "ldb = " << param.ldb << endl;
  cout << "ldc = " << param.ldc << endl;
  cout << "a_offset = " << param.a_offset << endl;
  cout << "b_offset = " << param.b_offset << endl;
  cout << "c_offset = " << param.c_offset << endl;
  cout << "a_stride = " << param.a_stride << endl;
  cout << "b_stride = " << param.b_stride << endl;
  cout << "c_stride = " << param.c_stride << endl;
  cout << "alpha = " << param.alpha << endl;
  cout << "bets = " << param.alpha << endl;
  cout << "batch_count = " << param.batch_count << endl;  
}

void dsaXInit(int dev){
#if DSA_XENGINE_TARGET_CUDA
  dsaXInitCuda(dev);
#endif
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

void dsaXCorrelator(void *output_data, void *input_data) {  

  dmem d;
  int bf = 0;
#if DSA_XENGINE_TARGET_CUDA
  initializeCudaMemory(&d, bf);
  d.h_input = (char *)malloc(sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  memcpy(d.h_input, (char*)input_data, sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  dcorrelator(&d);
  dsaXmemcpy(output_data, d.d_output, NBASE*NCHAN_PER_PACKET*2*2*4, dsaXMemcpyDeviceToHost);  
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
