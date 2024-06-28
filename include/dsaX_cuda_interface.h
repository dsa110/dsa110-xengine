#pragma once

#include <vector>

#include "dsaX_def.h"
#include "dsaX_enums.h"
#include "dsaX.h"

void dsaXInitCuda(int dev);

void initializeCorrCudaMemory(dmem_corr *d);

void initializeBFCudaMemory(dmem_bf *d);

void deallocateCorrCudaMemory(dmem_corr *d);

void deallocateBFCudaMemory(dmem_bf *d);

void dsaXmemsetCuda(void *array, int ch, size_t n);

void dsaXmemcpyCuda(void *array_device, void *array_host, size_t n, dsaXMemcpyKind kind);

void dsaXDeviceSynchronizeCuda();

void reorderCorrOutputCuda(dmem_corr *d);

void reorderCorrInputCuda(dmem_corr *d);

void calcWeightsCuda(dmem_bf *d);

template <typename in_prec, typename out_prec> void transposeMatrixCuda(in_prec *idata, out_prec *odata);

void transposeInputBeamformerCuda(double *idata, double *odata, std::vector<int> &dim_block_in, std::vector<int> &dim_grid_in);

void transposeScaleBeamformerCuda(void *real, void *imag, unsigned char *output, std::vector<int> &dim_block_in, std::vector<int> &dim_grid_in);

void fluffInputBeamformerCuda(char *input, void *b_real, void *b_imag, int blocks, int tpb);

void sumBeamCuda(unsigned char *input, float *output, int blocks, int tpb);
