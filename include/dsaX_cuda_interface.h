#pragma once

#include "dsaX_def.h"
#include "dsaX.h"

#ifdef DSA_XENGINE_TARGET_CUDA
void initialize_device_memory(dmem *d, int bf);

void deallocate_device_memory(dmem *d, int bf);

void reorder_output_device(dmem *d);

__global__ void corr_input_copy(char *input, half *inr, half *ini);

template <typename in_prec, typename out_prec> __global__ void transpose_matrix(in_prec *idata, out_prec *odata);

void reorder_input_device(char *input, char *tx, half *inr, half *ini);

__global__ void corr_output_copy(half *outr, half *outi, float *output, int *indices_lookup);

__global__ void transpose_input_bf(double *idata, double *odata);

__global__ void populate_weights_matrix(float *antpos_e, float *antpos_n, float *calibs, half *wr, half *wi, float *fqs);

void calc_weights(dmem *d);

__global__ void fluff_input_bf(char *input, half *dr, half *di);

__global__ void transpose_scale_bf(half *ir, half *ii, unsigned char *odata);

__global__ void sum_beam(unsigned char *input, float *output);
#endif
