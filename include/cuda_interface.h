#pragma once

#include <vector>

#include "dsaX_def.h"
#include "enums.h"
#include "dsaX.h"

void dsaXInitCuda(int dev);
void dsaXDestroyCuda();

void initBLASCuda();
void destroyBLASCuda();

void initStreamsCuda(unsigned int n);
void destroyStreamsCuda();

void promoteComplexCharToPlanarHalfCuda(corr_handle *d, unsigned int stream);

void initializeCorrCudaMemory(corr_handle *d, unsigned int n_streams);

void initializeBFCudaMemory(bf_handle *d);

void deallocateCorrCudaMemory(corr_handle *d);

void deallocateBFCudaMemory(bf_handle *d);

void dsaXmemsetCuda(void *array, int ch, size_t n);

void dsaXmemcpyCuda(void *array_device, void *array_host, size_t n, dsaXMemcpyKind kind, int stream);

void *dsaXHostRegisterCuda(size_t size);

void dsaXDeviceSynchronizeCuda();

void reorderCorrOutputCuda(corr_handle *d, int stream);

void computeIndicesCuda(corr_handle *d);

void reorderCorrInputCuda(corr_handle *d, int stream);

void calcWeightsCuda(bf_handle *d);

template <typename in_prec, typename out_prec> void transposeMatrixCuda(in_prec *idata, out_prec *odata);

void transposeInputBeamformerCuda(double *idata, double *odata, std::vector<int> &dim_block_in, std::vector<int> &dim_grid_in);

void transposeScaleBeamformerCuda(void *real, void *imag, unsigned char *output, std::vector<int> &dim_block_in, std::vector<int> &dim_grid_in);

void fluffInputBeamformerCuda(char *input, void *b_real, void *b_imag, int blocks, int tpb);

void sumBeamCuda(unsigned char *input, float *output, int blocks, int tpb);
