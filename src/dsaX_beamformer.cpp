// -*- c++ -*-
/* assumes input and output block size is appropriate - will seg fault otherwise*/
/*
Workflow is similar for BF and corr applications
 - copy data to GPU, convert to half-precision and calibrate while reordering
 - do matrix operations to populate large output vector
 */

#include <iostream>
#include <vector>

#include "dsaX_def.h"
#include "dsaX.h"
#include "dsaX_blas_interface.h"
#include "dsaX_utils.h"
#include "dsaX_psrdada_utils.h"

using namespace std;

/*
Beamformer:
 - initial data is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex] 
 - split into EW and NS antennas via cudaMemcpy: [NPACKETS_PER_BLOCK, NANTS/2, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
 - want [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, NANTS/2, 8chan, 2 times, 2 pol, 4-bit complex]
(single transpose operation)
 - weights are [NCHAN_PER_PACKET/8, NBEAMS, 4tim, NANTS/2, 8chan, 2 times, 2 pol] x 2
 - then fluff and run beamformer: output is [NCHAN_PER_PACKET/8, NBEAMS, NPACKETS_PER_BLOCK/4] (w column-major)
 - transpose and done! 

*/
// beamformer function
void dbeamformer(dmem_bf *d) {

  dsaXBLASParam blas_param;
  blas_param.trans_a = DSA_BLAS_OP_T;
  blas_param.trans_b = DSA_BLAS_OP_N;
  blas_param.m = NPACKETS_PER_BLOCK/4;
  blas_param.n = NBEAMS/2;
  blas_param.k = 4*(NANTS/2)*8*2*2;
  blas_param.alpha = 1.0;
  blas_param.lda = blas_param.k;
  blas_param.ldb = blas_param.k;
  blas_param.beta = 0.0;
  blas_param.ldc = blas_param.m;
  blas_param.a_stride = (NPACKETS_PER_BLOCK)*(NANTS/2)*8*2*2;
  blas_param.b_stride = (NBEAMS/2)*4*(NANTS/2)*8*2*2;
  blas_param.c_stride = (NPACKETS_PER_BLOCK/4)*NBEAMS/2;
  blas_param.batch_count = NCHAN_PER_PACKET/8;
  
  long long int i1, i2;
  
  // timing
  // copy, prepare, cublas, output
  clock_t begin, end;

  // do big memcpy
  begin = clock();
  dsaXmemcpy(d->d_big_input, d->h_input, NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4, dsaXMemcpyHostToDevice);
  end = clock();
  d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // loop over halves of the array
  for (int iArm=0;iArm<2;iArm++) {
  
    // zero out output arrays
    dsaXmemset(d->d_bigbeam_r,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(short));
    dsaXmemset(d->d_bigbeam_i,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(short));
    dsaXDeviceSynchronize();
    
    // copy data to device
    // initial data: [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
    // final data: need to split by NANTS.
    begin = clock();
    for (i1=0; i1<NPACKETS_PER_BLOCK; i1++) 
      dsaXmemcpy(d->d_input + i1*(NANTS/2)*NCHAN_PER_PACKET*4,
		 d->d_big_input + i1*(NANTS)*NCHAN_PER_PACKET*4+iArm*(NANTS/2)*NCHAN_PER_PACKET*4,
		 (NANTS/2)*NCHAN_PER_PACKET*4, dsaXMemcpyDeviceToDevice);
    end = clock();
    d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // do reorder and fluff of data to real and imag
    begin = clock();

    // DMH: Abstract the launch parameters
    std::vector<int> dimBlock = {16, 8};
    std::vector<int> dimGrid = {NCHAN_PER_PACKET/8/16, (NPACKETS_PER_BLOCK)*(NANTS/2)/16};
    transposeInputBeamformer((double *)(d->d_input), (double *)(d->d_tx), dimBlock, dimGrid);

    int blocks = NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128;
    int tpb = 128;
    fluffInputBeamformer(d->d_tx, d->d_br, d->d_bi, blocks, tpb);    
    end = clock();
    d->prep += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // set up for gemm    
    i2 = iArm*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8); // weights offset
    blas_param.b_offset = i2;
    // large matrix multiply to get real and imag outputs
    begin = clock();
    dsaXHgemmStridedBatched(d->d_br, d->d_bi, d->weights_r, d->weights_i, d->d_bigbeam_r, d->d_bigbeam_i, blas_param);
    end = clock();
    d->cubl += (float)(end - begin) / CLOCKS_PER_SEC;
        
    // simple formation of total power and scaling to 8-bit in transpose kernel
    // Reuse dimBlock
    //DMH: Abstract kernel launch parameters
    dimGrid[0] = (NBEAMS/2)*(NPACKETS_PER_BLOCK/4)/16;
    dimGrid[1] = (NCHAN_PER_PACKET/8)/16;
    begin = clock();
    transposeScaleBeamformer(d->d_bigbeam_r, d->d_bigbeam_i, d->d_bigpower + iArm*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2), dimBlock, dimGrid);
    end = clock();
    d->outp += (float)(end - begin) / CLOCKS_PER_SEC;
  }

  // form sum over times
  int blocks = 24576;
  int tpb = 512;
  // COMMENT OUT WHEN DONE!!!
  //sumBeam(d->d_bigpower, d->d_chscf, blocks, tpb);
}
