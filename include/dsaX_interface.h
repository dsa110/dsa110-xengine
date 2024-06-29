#pragma once

#include <vector>
#include "dsaX.h"

// DMH: decorate these with Doxygen
void dsaXCorrelator(void *input_data, void *output_data);

void reorderCorrInput(dmem_corr *d);

void reorderCorrOutput(dmem_corr *d);

void transposeInputBeamformer(double *input, double *output, std::vector<int> &dimBlock, std::vector<int> &dimGrid);

void transposeScaleBeamformer(void *array_real, void *array_imag, unsigned char *output, std::vector<int> &dimBlock, std::vector<int> &dimGrid);

void fluffInputBeamformer(char *input, void *array_real, void *array_imag, int blocks, int tpb);

void sumBeam(unsigned char *input, float *output, int blocks, int tpb);
