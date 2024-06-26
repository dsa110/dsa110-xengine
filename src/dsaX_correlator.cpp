// -*- c++ -*-
/* assumes input and output block size is appropriate - will seg fault otherwise*/
/*
Workflow is similar for BF and corr applications
 - copy data to GPU, convert to half-precision and calibrate while reordering
 - do matrix operations to populate large output vector
 */

#include <iostream>

#include "dsaX_def.h"
#include "dsaX.h"
#include "dsaX_blas_interface.h"
#include "dsaX_utils.h"
#include "dsaX_psrdada_utils.h"

// correlator function
// workflow: copy to device, reorder, stridedBatchedGemm, reorder
// DMH CUDA references excised.
void dcorrelator(dmem *d) {

  // copy to device
  dsaXmemcpy(d->d_input, d->h_input, NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2, dsaXMemcpyHostToDevice);
  
  // zero out output arrays
  dsaXmemset(d->d_outr, 0, NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(short)); //half -> short
  dsaXmemset(d->d_outi, 0, NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(short)); //half -> short
  dsaXmemset(d->d_output, 0, NCHAN_PER_PACKET*2*NANTS*NANTS*sizeof(float));
  
  // reorder input into real and imaginary arrays of 2 byte data
  reorderInput(d);

  dsaXBLASParam blas_param;
  // gemm settings
  // input: [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
  // output: [NCHAN_PER_PACKET, 2times, 2pol, NANTS, NANTS] 
  blas_param.trans_a = DSA_BLAS_OP_N;
  blas_param.trans_b = DSA_BLAS_OP_T;
  blas_param.m = NANTS;
  blas_param.n = NANTS;
  blas_param.k = NPACKETS_PER_BLOCK/halfFac;
  blas_param.alpha = 1.0;
  blas_param.lda = blas_param.m;
  blas_param.ldb = blas_param.n;
  blas_param.beta = 0.;
  blas_param.ldc = blas_param.m;
  blas_param.a_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;
  blas_param.b_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;
  blas_param.c_stride = NANTS*NANTS;
  blas_param.batch_count = NCHAN_PER_PACKET*2*2*halfFac;

  // Perform GEMM accoring to back end configuration
  dsaXHgemmStridedBatched(d->d_r, d->d_i, d->d_r, d->d_i, d->d_outr, d->d_outi, blas_param);
  
  // reorder output data
  reorderOutput(d);
}
