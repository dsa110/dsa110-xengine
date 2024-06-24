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
#ifdef DSA_XENGINE_TARGET_CUDA
#include "dsaX_cuda_interface.h"
#endif

using namespace std;

int DEBUG = 1;

void usage() {
  fprintf (stdout,
	   "dsaX_beamformer_correlator [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default REORDER_BLOCK_KEY]\n"
	   " -o out_key [default XGPU_BLOCK_KEY]\n"
	   " -b run beamformer [default is to run correlator]\n"
	   " -h print usage\n"
	   " -t binary file for test mode\n"
	   " -f flagants file\n"
	   " -a calib file\n"
	   " -s start frequency (assumes -0.244140625MHz BW)\n");
}


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
void dbeamformer(dmem *d) {

  // gemm settings - recall column major order assumed
  // stride over 48 chans
  cublasHandle_t cublasH = NULL;
  cublasCreate(&cublasH);
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = NPACKETS_PER_BLOCK/4;
  const int n = NBEAMS/2;
  const int k = 4*(NANTS/2)*8*2*2;
  const half alpha = 1.;
  const half malpha = -1.;
  const int lda = k;
  const int ldb = k;
  const half beta0 = 0.;
  const half beta1 = 1.;
  const int ldc = m;
  const long long int strideA = (NPACKETS_PER_BLOCK)*(NANTS/2)*8*2*2;
  const long long int strideB = (NBEAMS/2)*4*(NANTS/2)*8*2*2;
  const long long int strideC = (NPACKETS_PER_BLOCK/4)*NBEAMS/2;
  const int batchCount = NCHAN_PER_PACKET/8;
  long long int i1, i2;//, o1;
  
  // create streams
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // timing
  // copy, prepare, cublas, output
  clock_t begin, end;

  // do big memcpy
  begin = clock();
  dsaXmemcpyHostToDevice(d->d_big_input,d->h_input,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4);
  end = clock();
  d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // loop over halves of the array
  for (int iArm=0;iArm<2;iArm++) {
  
    // zero out output arrays
    dsaXmemset(d->d_bigbeam_r,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(half));
    dsaXmemset(d->d_bigbeam_i,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(half));
    cudaDeviceSynchronize();
    
    // copy data to device
    // initial data: [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
    // final data: need to split by NANTS.
    begin = clock();
    for (i1=0; i1<NPACKETS_PER_BLOCK; i1++) 
      dsaXmemcpyDeviceToDevice(d->d_input+i1*(NANTS/2)*NCHAN_PER_PACKET*4,
			       d->d_big_input+i1*(NANTS)*NCHAN_PER_PACKET*4+iArm*(NANTS/2)*NCHAN_PER_PACKET*4,
			       (NANTS/2)*NCHAN_PER_PACKET*4);
    end = clock();
    d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // do reorder and fluff of data to real and imag
    begin = clock();
    
    dim3 dimBlock1(16, 8), dimGrid1(NCHAN_PER_PACKET/8/16, (NPACKETS_PER_BLOCK)*(NANTS/2)/16);    
    transpose_input_bf<<< dimGrid1, dimBlock1 >>>((double *)(d->d_input), (double *)(d->d_tx));    
    fluff_input_bf<<<NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128,128>>>(d->d_tx, d->d_br, d->d_bi);
    
    end = clock();
    d->prep += (float)(end - begin) / CLOCKS_PER_SEC;

    // large matrix multiply to get real and imag outputs
    // set up for gemm
    cublasSetStream(cublasH, stream);
    i2 = iArm*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8); // weights offset
    
    // run strided batched gemm
    begin = clock();
    // ac
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_br,lda,strideA,
			      d->weights_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_r,ldc,strideC,
			      batchCount);
    // -bd
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &malpha,d->d_bi,lda,strideA,
			      d->weights_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_r,ldc,strideC,
			      batchCount);
    // bc
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bi,lda,strideA,
			      d->weights_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_i,ldc,strideC,
			      batchCount);
    // ad
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_br,lda,strideA,
			      d->weights_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_i,ldc,strideC,
			      batchCount);
      
    cudaDeviceSynchronize();
    end = clock();
    d->cubl += (float)(end - begin) / CLOCKS_PER_SEC;
        
    // simple formation of total power and scaling to 8-bit in transpose kernel
    begin = clock();
    dim3 dimBlock(16, 8), dimGrid((NBEAMS/2)*(NPACKETS_PER_BLOCK/4)/16, (NCHAN_PER_PACKET/8)/16);
    transpose_scale_bf<<<dimGrid,dimBlock>>>(d->d_bigbeam_r,d->d_bigbeam_i,d->d_bigpower+iArm*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2));
    end = clock();
    d->outp += (float)(end - begin) / CLOCKS_PER_SEC;
  }

  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);

  // form sum over times
  //sum_beam<<<24576,512>>>(d->d_bigpower,d->d_chscf);
}
