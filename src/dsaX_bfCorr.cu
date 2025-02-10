// -*- c++ -*-
/* assumes input and output block size is appropriate - will seg fault otherwise*/
/*
Workflow is similar for BF and corr applications
 - copy data to GPU, convert to half-precision and calibrate while reordering
 - do matrix operations to populate large output vector
 */
#include <iostream>
#include <algorithm>
using std::cout;
using std::cerr;
using std::endl;
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>
#include <syslog.h>
#include <pthread.h>

#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_def.h"

#include <cuda.h>
#include "cuda_fp16.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// required to prevent overflow in corr matrix multiply
#define halfFac 4

// cycle on which total powers are recorded
#define power_cycle 8

// beam sep
#define sep 1.0 // arcmin
#define sep_ns 0.75 // arcmin

/* global variables */
int DEBUG = 0;

// define structure that carries around device memory
typedef struct dmem {

  // initial data and streams
  char * h_input, * h_pinned_input; // host input pointer
  char * d_input, * d_tx; // [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
  
  // correlator pointers
  // giant array for r and i: [NCHAN_PER_PACKET, 2 pol, NANTS_PROCESS, NPACKETS_PER_BLOCK * 2 times]
  half * d_r, * d_i;
  // arrays for matrix multiply output: input [NANTS_PROCESS, NANTS_PROCESS]
  half * d_outr, *d_outi, *d_htx, *d_tx_outr, *d_tx_outi;
  // giant output array: [NBASE, NCHAN_PER_PACKET, 2 pol, 2 complex]
  float * d_output;
  
  // beamformer pointers
  char * d_big_input;
  half * d_bar, * d_bai, * d_bbr, * d_bbi;
  half * d_ibsum;
  half * weights_a_r, * weights_a_i, * weights_b_r, * weights_b_i; 
  half * d_bigbeam_a_r, * d_bigbeam_a_i, * d_bigbeam_b_r, * d_bigbeam_b_i; 
  unsigned char * d_bigpower; 
  float * d_scf; // scale factor per beam
  float * d_chscf, * h_chscf;
  float * h_winp;
  int * flagants, nflags;
  int * d_flagants;
  float * h_freqs, * d_freqs;
  int subtract_ib;

  // timing
  float cp, prep, cubl, outp;

  // obs dec
  float obsdec;
  
} dmem;

/*! register the data_block in the hdu via cudaHostRegister */
int dada_cuda_dbregister (dada_hdu_t * hdu)
{
  ipcbuf_t * db = (ipcbuf_t *) hdu->data_block;

  // ensure that the data blocks are SHM locked
  if (ipcbuf_lock (db) < 0)
  {
    syslog(LOG_ERR,"dada_dbregister: ipcbuf_lock failed");
    return -1;
  }

  size_t bufsz = db->sync->bufsz;
  unsigned int flags = 0;
  cudaError_t rval;

  // lock each data block buffer as cuda memory
  uint64_t ibuf;
  for (ibuf = 0; ibuf < db->sync->nbufs; ibuf++)
  {
    rval = cudaHostRegister ((void *) db->buffer[ibuf], bufsz, flags);
    if (rval != cudaSuccess)
    {
      syslog(LOG_ERR,"dada_dbregister:  cudaHostRegister failed");
      return -1;
    }
  }
  
  return 0;
}


// allocate device memory
void initialize(dmem * d, int bf, int subtract_ib) {
  
  // for correlator
  if (bf==0) {
    cudaMallocHost((void**)&d->h_pinned_input, sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_input), sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_r), sizeof(half)*NCHAN_PER_PACKET*2*NANTS*NPACKETS_PER_BLOCK*2);
    cudaMalloc((void **)(&d->d_i), sizeof(half)*NCHAN_PER_PACKET*2*NANTS*NPACKETS_PER_BLOCK*2);
    cudaMalloc((void **)(&d->d_tx), sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_output), sizeof(float)*NBASE*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_outr), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
    cudaMalloc((void **)(&d->d_outi), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
    cudaMalloc((void **)(&d->d_tx_outr), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
    cudaMalloc((void **)(&d->d_tx_outi), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);

    // timers
    d->cp = 0.;
    d->prep = 0.;
    d->outp = 0.;
    d->cubl = 0.;
    
  }

  // for beamformer
  if (bf==1) {
    cudaMalloc((void **)(&d->d_input), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS/2)*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_big_input), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS)*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_htx), sizeof(half)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2);
    cudaMalloc((void **)(&d->d_ibsum), sizeof(half)*(NCHAN_PER_PACKET/8)*8*2*NPACKETS_PER_BLOCK);
    cudaMalloc((void **)(&d->d_bar), sizeof(half)*(NCHAN_PER_PACKET/8)*8*2*NPACKETS_PER_BLOCK*(NANTS/2));
    cudaMalloc((void **)(&d->d_bai), sizeof(half)*(NCHAN_PER_PACKET/8)*8*2*NPACKETS_PER_BLOCK*(NANTS/2));
    cudaMalloc((void **)(&d->d_bbr), sizeof(half)*(NCHAN_PER_PACKET/8)*8*2*NPACKETS_PER_BLOCK*(NANTS/2));
    cudaMalloc((void **)(&d->d_bbi), sizeof(half)*(NCHAN_PER_PACKET/8)*8*2*NPACKETS_PER_BLOCK*(NANTS/2));
    cudaMalloc((void **)(&d->weights_a_r), sizeof(half)*2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2));
    cudaMalloc((void **)(&d->weights_a_i), sizeof(half)*2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2));
    cudaMalloc((void **)(&d->weights_b_r), sizeof(half)*2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2));
    cudaMalloc((void **)(&d->weights_b_i), sizeof(half)*2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2));
    cudaMalloc((void **)(&d->d_bigbeam_a_r), sizeof(half)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK);
    cudaMalloc((void **)(&d->d_bigbeam_a_i), sizeof(half)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK);
    cudaMalloc((void **)(&d->d_bigbeam_b_r), sizeof(half)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK);
    cudaMalloc((void **)(&d->d_bigbeam_b_i), sizeof(half)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK);
    cudaMalloc((void **)(&d->d_bigpower), sizeof(unsigned char)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS));
    cudaMalloc((void **)(&d->d_chscf), sizeof(float)*NBEAMS); // beam scale factor
    cudaMalloc((void **)(&d->d_flagants), sizeof(int)*NANTS); // flag ants
    d->h_chscf = (float *)malloc(sizeof(float)*NBEAMS);
    
    // input weights: first is [NANTS, E/N], then [NANTS, 48, 2pol, R/I]
    d->h_winp = (float *)malloc(sizeof(float)*(NANTS*2+NANTS*(NCHAN_PER_PACKET/8)*2*2));
    d->flagants = (int *)malloc(sizeof(int)*NANTS);
    d->h_freqs = (float *)malloc(sizeof(float)*(NCHAN_PER_PACKET/8));
    cudaMalloc((void **)(&d->d_freqs), sizeof(float)*(NCHAN_PER_PACKET/8));

    // timers
    d->cp = 0.;
    d->prep = 0.;
    d->outp = 0.;
    d->cubl = 0.;
    
  }

  // subtract_ib
  d->subtract_ib = subtract_ib;
  
}

// deallocate device memory
void deallocate(dmem * d, int bf) {

  cudaFree(d->d_input);

  if (bf==0) {
    cudaFree(d->d_r);
    cudaFree(d->d_i);
    cudaFree(d->d_tx);
    cudaFree(d->d_output);
    cudaFree(d->d_outr);
    cudaFree(d->d_outi);
    cudaFree(d->d_tx_outr);
    cudaFree(d->d_tx_outi);
    cudaFreeHost(d->h_pinned_input);
  }
  if (bf==1) {
    cudaFree(d->d_htx);
    cudaFree(d->d_ibsum);
    cudaFree(d->d_bar);
    cudaFree(d->d_bai);
    cudaFree(d->d_bbr);
    cudaFree(d->d_bbi);
    cudaFree(d->weights_a_r);
    cudaFree(d->weights_a_i);
    cudaFree(d->weights_b_r);
    cudaFree(d->weights_b_i);
    cudaFree(d->d_bigbeam_a_r);
    cudaFree(d->d_bigbeam_a_i);
    cudaFree(d->d_bigbeam_b_r);
    cudaFree(d->d_bigbeam_b_i);
    cudaFree(d->d_bigpower);
    //cudaFree(d->d_scf);
    cudaFree(d->d_chscf);
    free(d->h_winp);
    free(d->flagants);
    cudaFree(d->d_freqs);
    free(d->h_freqs);
  }
  
}

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);
int dada_bind_thread_to_core (int core);

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out)
{

  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);

  if (dada_hdu_unlock_write (out) < 0)
    {
      syslog(LOG_ERR, "could not unlock write on hdu_out");
    }
  dada_hdu_destroy (out);

} 


void usage()
{
fprintf (stdout,
	 "dsaX_bfCorr [options]\n"
	 " -c core   bind process to CPU core [no default]\n"
	 " -d send debug messages to syslog\n"
	 " -i in_key [default REORDER_BLOCK_KEY]\n"
	 " -o out_key [default XGPU_BLOCK_KEY]\n"
	 " -b run beamformer [default is to run correlator]\n"
	 " -h print usage\n"
	 " -t binary file for test mode\n"
	 " -f flagants file\n"
	 " -a calib file\n"
	 " -s start frequency (assumes -0.244140625MHz BW)\n"
	 " -g observing DEC in degrees (default 71.66)\n"
	 " -p full path of beam powers file (default powers.out)\n"
	 " -k subtract incoherent beam\n");
}

// kernel to fluff input
// run with 128 threads and NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/128 blocks
__global__ void corr_input_copy(char *input, half *inr, half *ini) {

  int bidx = blockIdx.x; // assume NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/128
  int tidx = threadIdx.x; // assume 128
  int iidx = bidx*128+tidx;
  
  inr[iidx] = __float2half((float)((char)(((unsigned char)(input[iidx]) & (unsigned char)(15)) << 4) >> 4));
  ini[iidx] = __float2half((float)((char)(((unsigned char)(input[iidx]) & (unsigned char)(240))) >> 4));

}


// arbitrary transpose kernel
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width is the dimension of the fastest index
__global__ void transpose_matrix_char(char * idata, char * odata) {

  __shared__ char tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

}

// arbitrary transpose kernel
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width is the dimension of the fastest index
__global__ void transpose_matrix_float(half * idata, half * odata) {

  __shared__ half tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

}


// function to copy amd reorder d_input to d_r and d_i
// input is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
// output is [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
// starts by running transpose on [NPACKETS_PER_BLOCK * NANTS, NCHAN_PER_PACKET * 2 * 2] matrix in doubleComplex form.
// then fluffs using simple kernel
void reorder_input(char *input, char * tx, half *inr, half *ini) {

  // transpose input data
  dim3 dimBlock(32, 8), dimGrid((NCHAN_PER_PACKET*2*2)/32, ((NPACKETS_PER_BLOCK)*NANTS)/32);
  transpose_matrix_char<<<dimGrid,dimBlock>>>(input,tx);
  /*
  // set up for geam
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublasH, stream);

  // transpose input matrix into tx
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = NPACKETS_PER_BLOCK * NANTS;
  const int n = NCHAN_PER_PACKET*2*2/8; // columns in output
  const double alpha = 1.0;
  const double beta = 0.0;
  const int lda = n;
  const int ldb = m;
  const int ldc = ldb;
  cublasDgeam(cublasH,transa,transb,m,n,
	      &alpha,(double *)(input),
	      lda,&beta,(double *)(tx),
	      ldb,(double *)(tx),ldc);
  */
  // now we just need to fluff to half-precision
  corr_input_copy<<<NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/128,128>>>(tx,inr,ini);

  // look at output
  /*char * odata = (char *)malloc(sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4*2);
  cudaMemcpy(odata,inr,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4*2,cudaMemcpyDeviceToHost);
  FILE *fout;
  fout=fopen("test.test","wb");
  fwrite(odata,1,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4*2,fout);
  fclose(fout);*/
  
  // destroy stream
  //cudaStreamDestroy(stream);
  
}

// kernel to help with reordering output
// outr and outi are [NANTS, NANTS, NCHAN_PER_PACKET, 2time, 2pol, halfFac]
// run with NCHAN_PER_PACKET*2*NBASE/128 blocks of 128 threads
__global__ void corr_output_copy(half *outr, half *outi, float *output, int *indices_lookup) {

  int bidx = blockIdx.x; // assume NCHAN_PER_PACKET*2*NBASE/128
  int tidx = threadIdx.x; // assume 128
  int idx = bidx*128+tidx;
  
  int baseline = (int)(idx / (NCHAN_PER_PACKET * 2));
  int chpol = (int)(idx % (NCHAN_PER_PACKET * 2));
  int ch = (int)(chpol / 2);
  int base_idx = indices_lookup[baseline];
  int iidx = base_idx * NCHAN_PER_PACKET + ch;
  int pol = (int)(chpol % 2);

  float v1=0., v2=0.;
  
  for (int i=0;i<halfFac;i++) {
    v1 += __half2float(outr[(4*iidx+pol)*halfFac+i])+__half2float(outr[(4*iidx+2+pol)*halfFac+i]);
    v2 += __half2float(outi[(4*iidx+pol)*halfFac+i])+__half2float(outi[(4*iidx+2+pol)*halfFac+i]);
  }

  output[2*idx] = v1;
  output[2*idx+1] = v2;
  
}


// function to copy d_outr and d_outi to d_output
// inputs are [NCHAN_PER_PACKET, 2 time, 2 pol, NANTS, NANTS]
// the corr matrices are column major order
// output needs to be [NBASE, NCHAN_PER_PACKET, 2 pol, 2 complex]
// start with transpose to get [NANTS*NANTS, NCHAN_PER_PACKET*2*2], then sum into output using kernel
void reorder_output(dmem * d) {

  // transpose input data
  dim3 dimBlock(32, 8), dimGrid((NANTS*NANTS)/32,(NCHAN_PER_PACKET*2*2*halfFac)/32);
  transpose_matrix_float<<<dimGrid,dimBlock>>>(d->d_outr,d->d_tx_outr);
  transpose_matrix_float<<<dimGrid,dimBlock>>>(d->d_outi,d->d_tx_outi);

  // look at output
  /*char * odata = (char *)malloc(sizeof(char)*384*4*NANTS*NANTS*2*halfFac);
  cudaMemcpy(odata,d->d_tx_outr,384*4*NANTS*NANTS*2*halfFac,cudaMemcpyDeviceToHost);
  FILE *fout;
  fout=fopen("test2.test","wb");
  fwrite(odata,sizeof(char),384*4*NANTS*NANTS*2*halfFac,fout);
  fclose(fout);*/

  
  /*
  // set up for geam
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublasH, stream);

  // transpose output matrices into tx_outr and tx_outi
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = NCHAN_PER_PACKET*2*2;
  const int n = NANTS*NANTS/16; // columns in output
  const double alpha = 1.0;
  const double beta = 0.0;
  const int lda = n;
  const int ldb = m;
  const int ldc = ldb;
  cublasDgeam(cublasH,transa,transb,m,n,
	      &alpha,(double *)(d->d_outr),
	      lda,&beta,(double *)(d->d_tx_outr),
	      ldb,(double *)(d->d_tx_outr),ldc);
  cublasDgeam(cublasH,transa,transb,m,n,
	      &alpha,(double *)(d->d_outi),
	      lda,&beta,(double *)(d->d_tx_outi),
	      ldb,(double *)(d->d_tx_outi),ldc);
  */
  // now run kernel to sum into output
  int * h_idxs = (int *)malloc(sizeof(int)*NBASE);
  int * d_idxs;
  cudaMalloc((void **)(&d_idxs), sizeof(int)*NBASE);
  int ii = 0;
  // upper triangular order (column major) to match xGPU (not the same as CASA!)
  for (int i=0;i<NANTS;i++) {
    for (int j=0;j<=i;j++) {
      h_idxs[ii] = i*NANTS + j;
      ii++;
    }
  }
  cudaMemcpy(d_idxs,h_idxs,sizeof(int)*NBASE,cudaMemcpyHostToDevice);

  // run kernel to finish things
  corr_output_copy<<<NCHAN_PER_PACKET*2*NBASE/128,128>>>(d->d_tx_outr,d->d_tx_outi,d->d_output,d_idxs);

  /*char * odata = (char *)malloc(sizeof(char)*384*4*NBASE*4);
  cudaMemcpy(odata,d->d_output,384*4*NBASE*4,cudaMemcpyDeviceToHost);
  FILE *fout;
  fout=fopen("test3.test","wb");
  fwrite(odata,sizeof(char),384*4*NBASE*4,fout);
  fclose(fout);*/

  
  cudaFree(d_idxs);
  free(h_idxs);
  //cudaStreamDestroy(stream);  

}



// correlator function
// workflow: copy to device, reorder, stridedBatchedGemm, reorder
void dcorrelator(dmem * d) {

  // timing
  // copy, prepare, cublas, output
  clock_t begin, end;

  // zero out output arrays
  cudaMemset(d->d_outr,0,NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(half));
  cudaMemset(d->d_outi,0,NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(half));
  cudaMemset(d->d_output,0,NCHAN_PER_PACKET*2*NANTS*NANTS*sizeof(float));
  
  // copy to device  
  //memcpy(d->h_pinned_input,d->h_input,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  begin = clock();
  cudaMemcpy(d->d_input,d->h_input,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2,cudaMemcpyHostToDevice);
  end = clock();
  d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // reorder input
  begin = clock();
  reorder_input(d->d_input,d->d_tx,d->d_r,d->d_i);
  
  // not sure if essential
  cudaDeviceSynchronize();
  end = clock();
  d->prep += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // set up for gemm

  begin = clock();
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  // gemm settings
  // input: [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
  // output: [NCHAN_PER_PACKET, 2times, 2pol, NANTS, NANTS] 
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  const int m = NANTS;
  const int n = NANTS;
  const int k = NPACKETS_PER_BLOCK/halfFac;
  const half alpha = 1.;
  const half malpha = -1.;
  const int lda = m;
  const int ldb = n;
  const half beta0 = 0.;
  const half beta1 = 1.;
  const int ldc = m;
  const long long int strideA = NPACKETS_PER_BLOCK*NANTS/halfFac;
  const long long int strideB = NPACKETS_PER_BLOCK*NANTS/halfFac;
  const long long int strideC = NANTS*NANTS;
  const int batchCount = NCHAN_PER_PACKET*2*2*halfFac;

  // run strided batched gemm
  // ac
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,d->d_r,lda,strideA,
			    d->d_r,ldb,strideB,&beta0,
			    d->d_outr,ldc,strideC,
			    batchCount);
  // bd
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,d->d_i,lda,strideA,
			    d->d_i,ldb,strideB,&beta1,
			    d->d_outr,ldc,strideC,
			    batchCount);
  // -bc
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &malpha,d->d_i,lda,strideA,
			    d->d_r,ldb,strideB,&beta0,
			    d->d_outi,ldc,strideC,
			    batchCount);
  // ad
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,d->d_r,lda,strideA,
			    d->d_i,ldb,strideB,&beta1,
			    d->d_outi,ldc,strideC,
			    batchCount);

  // shown to be essential
  cudaDeviceSynchronize();
  end = clock();
  d->cubl += (float)(end - begin) / CLOCKS_PER_SEC;

  // destroy stream
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  
  // reorder output data
  begin = clock();
  reorder_output(d);
  end = clock();
  d->outp += (float)(end - begin) / CLOCKS_PER_SEC;
  
}

// kernels to reorder and fluff data for beamformer
// initial data is [NPACKETS_PER_BLOCK, (NANTS/2), NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]

/* TRANSPOSE AND SCALE INPUT
 - Input is [NPACKETS_PER_BLOCK, NANTS/2, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
 - want [NCHAN_PER_PACKET/8, 8chan, 2 times, NPACKETS_PER_BLOCK, NANTS/2] per pol and complexity
 - Do a 2-byte transpose, then fluff out data into four outputs
*/
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width=NCHAN_PER_PACKET*2, height=(NPACKETS_PER_BLOCK)*(NANTS/2)
__global__ void transpose_fluff_bf(unsigned short * idata, half * dra, half * dia, half * drb, half * dib) {

  // start with transpose
  
  __shared__ unsigned short tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8) 
    tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
  
  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  int oidx;
  unsigned short oval;
  unsigned char tmp;
  
  for (int j = 0; j < 32; j += 8) {
    //odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

    // output details
    oidx = (y+j)*width + x;
    oval = tile[threadIdx.x][threadIdx.y + j];

    // do casting to extract real/imag parts
    tmp = oval & 0xFF;
    dra[oidx] = __float2half(0.05*((float)((char)((tmp & (unsigned char)(15)) << 4) >> 4)));
    dia[oidx] = __float2half(0.05*((float)((char)((tmp & (unsigned char)(240))) >> 4)));
    
    tmp = (oval >> 8) & 0xFF;
    drb[oidx] = __float2half(0.05*((float)((char)((tmp & (unsigned char)(15)) << 4) >> 4)));
    dib[oidx] = __float2half(0.05*((float)((char)((tmp & (unsigned char)(240))) >> 4)));
        
  }


}

/* POWER SUM AND TRANSPOSE OUTPUT
 - Input for each pol and r/i is [NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times, NPACKETS_PER_BLOCK] 
 - want to form total power, and sum total powers over 4 PACKETS_PER_BLOCK
 - Then do a transpose to [NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times]
 - if doing subtract_ib:
  + ibsum has shape [NCHAN_PER_PACKET/8, 8chan, 2 times, NPACKETS_PER_BLOCK] 
  + 
*/
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width=NPACKETS_PER_BLOCK/4, height=NCHAN_PER_PACKET/8 * NBEAMS/2 * 8chan * 2times
__global__ void power_sum_and_transpose_output(half * dra, half * drb, half * dia, half * dib, half * ibsum, int subtract_ib, half * outp) {

  __shared__ half tile[32][33];

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  int iidx, iChan, iBeamSlow, iFast, idx;
  
  for (int j = 0; j < 32; j += 8) {

    iidx = (y+j)*width + x;
    iChan = (int)(iidx / ((NBEAMS/2)*8*2*NPACKETS_PER_BLOCK/4));
    iBeamSlow = (int)(iidx % ((NBEAMS/2)*8*2*NPACKETS_PER_BLOCK/4));
    iFast = (int)(iBeamSlow % (8*2*NPACKETS_PER_BLOCK/4));
    idx = iChan*8*2*NPACKETS_PER_BLOCK/4 + iFast;
    
    tile[threadIdx.y+j][threadIdx.x] = 0.;
    
    // do power sum
    for (int k=0;k<4;k++) {
      tile[threadIdx.y+j][threadIdx.x] += dra[4*iidx+k]*dra[4*iidx+k] + dia[4*iidx+k]*dia[4*iidx+k] + drb[4*iidx+k]*drb[4*iidx+k] + dib[4*iidx+k]*dib[4*iidx+k];
      if (subtract_ib)
	tile[threadIdx.y+j][threadIdx.x] -= ibsum[4*idx+k];
    }
      
  }
    
  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8) 
    outp[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];        

}

/* SUM TRANSPOSE AND SCALE OUTPUT
 - Input is [NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times] 
 - want to sum over 8 chan and 2 times
 - Then do a transpose to [NBEAMS, NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8]
*/
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width=NBEAMS/2, height=NPACKETS_PER_BLOCK/4 * NCHAN_PER_PACKET/8
__global__ void sum_transpose_and_scale_output(half * outp, unsigned char * odata, int subtract_ib) {

  __shared__ float tile[32][33];

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  int iidx;
  
  for (int j = 0; j < 32; j += 8) {
    
    iidx = (y+j)*width + x;
    tile[threadIdx.y+j][threadIdx.x] = 0.;
    
    // do sum over 8 chan and 2 times
    for (int k=0;k<16;k++) 
      tile[threadIdx.y+j][threadIdx.x] += __half2float(outp[16*iidx + k]);
      
  }
    
  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8) {
    if (subtract_ib==0) 
      odata[(y+j)*width + x] = (unsigned char)(tile[threadIdx.x][threadIdx.y + j]);
    else
      odata[(y+j)*width + x] = (unsigned char)(70.+tile[threadIdx.x][threadIdx.y + j]);
  }

}


// sum over all times and channels in output beam array
// run with NBEAMS blocks of 512 threads
__global__ void sum_beam(unsigned char * input, float * output) {

  extern __shared__ float psum[512];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int npartials = 48; // number partial sums

  int idx0 = bid*512*48 + tid*48;
  psum[tid] = 0.;
  for (int i=idx0;i<npartials+idx0;i++)
    psum[tid] += (float)(input[i]);

  __syncthreads();

  // sum over shared memory
  if (tid < 256) { psum[tid] += psum[tid + 256]; } __syncthreads(); 
  if (tid < 128) { psum[tid] += psum[tid + 128]; } __syncthreads(); 
  if (tid < 64) { psum[tid] += psum[tid + 64]; } __syncthreads();
  if (tid < 32) { psum[tid] += psum[tid + 32]; } __syncthreads();
  if (tid < 16) { psum[tid] += psum[tid + 16]; } __syncthreads();
  if (tid < 8) { psum[tid] += psum[tid + 8]; } __syncthreads();
  if (tid < 4) { psum[tid] += psum[tid + 4]; } __syncthreads();
  if (tid < 2) { psum[tid] += psum[tid + 2]; } __syncthreads();
  if (tid < 1) { psum[tid] += psum[tid + 1]; } __syncthreads(); 

  __syncthreads();

  if (tid==0) output[bid] = psum[0]/512./48.;
  
}


// sum over all powers of all antennas in input voltage array, removing flagged ones
// also sum over pols
// input is [NCHAN_PER_PACKET/8, 8chan, 2 times, NPACKETS_PER_BLOCK, NANTS/2]
// run with NCHAN_PER_PACKET*2*NPACKETS_PER_BLOCK blocks of 32 threads
__global__ void sum_ib(half * dra, half * dia, half * drb, half * dib, half * dout, int * flagants) {

  extern __shared__ half ppsum[32];
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*48 + tid;
  ppsum[tid] = 0.;
  if (flagants[tid]==0)
    ppsum[tid] = dra[idx]*dra[idx] + dia[idx]*dia[idx] + drb[idx]*drb[idx] + dib[idx]*dib[idx];
  __syncthreads();

  if (tid < 16) {
    idx = bid*48 + tid + 32;
    if (flagants[tid+32] == 0.)
      ppsum[tid] += dra[idx]*dra[idx] + dia[idx]*dia[idx] + drb[idx]*drb[idx] + dib[idx]*dib[idx];
  }

  __syncthreads();

  // sum over shared memory
  if (tid < 16) { ppsum[tid] += ppsum[tid + 16]; } __syncthreads();
  if (tid < 8) { ppsum[tid] += ppsum[tid + 8]; } __syncthreads();
  if (tid < 4) { ppsum[tid] += ppsum[tid + 4]; } __syncthreads();
  if (tid < 2) { ppsum[tid] += ppsum[tid + 2]; } __syncthreads();
  if (tid < 1) { ppsum[tid] += ppsum[tid + 1]; } __syncthreads(); 

  __syncthreads();

  if (tid==0) dout[bid] = ppsum[0];
  
}


/*
Beamformer:
 - initial data is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex] 
 - split into EW and NS antennas via cudaMemcpy: [NPACKETS_PER_BLOCK, NANTS/2, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
 - want [NCHAN_PER_PACKET/8, 8chan, 2 times, NPACKETS_PER_BLOCK, NANTS/2] for each pol and r/i
 - this is a simple 2-byte transpose and a memcpy after fluffing
 - weights can now be [NCHAN_PER_PACKET/8, NBEAMS/2, NANTS/2] for each pol and r/i, and arm

transpose of input gives m=8chan*2times*NPACKETS_PER_BLOCK, k = NANTS/2.
weights already have k = NANTS/2, n=NBEAMS/2.
output has m as fastest axis, and n as slowest axis (i.e., column major order)
so output of batched matrix mult is [NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times, NPACKETS_PER_BLOCK] 

 - can transform to output with two sum-and-transpose operations: [NBEAMS/2, NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8]. The first needs to form total power

OLD SCHEME
 - initial data is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex] 
 - split into EW and NS antennas via cudaMemcpy: [NPACKETS_PER_BLOCK, NANTS/2, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
 - want [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, NANTS/2, 8chan, 2 times, 2 pol, 4-bit complex]
(single transpose operation)
 - weights are [NCHAN_PER_PACKET/8, NBEAMS, 4tim, NANTS/2, 8chan, 2 times, 2 pol] x 2
 - then fluff and run beamformer: output is [NCHAN_PER_PACKET/8, NBEAMS, NPACKETS_PER_BLOCK/4] (w column-major)
 - transpose and done! 

*/
// beamformer function
void dbeamformer(dmem * d) {

  // gemm settings - recall column major order assumed
  cublasHandle_t cublasH = NULL;
  cublasCreate(&cublasH);
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = 8*2*NPACKETS_PER_BLOCK;
  const int n = NBEAMS/2;
  const int k = NANTS/2;
  const half alpha = 1.;
  const half malpha = -1.;
  const int lda = k;
  const int ldb = k;
  const half beta0 = 0.;
  const half beta1 = 1.;
  const int ldc = m;
  const long long int strideA = 8*2*NPACKETS_PER_BLOCK*(NANTS/2);
  const long long int strideB = (NBEAMS/2)*(NANTS/2);
  const long long int strideC = (NBEAMS/2)*8*2*NPACKETS_PER_BLOCK;
  const int batchCount = NCHAN_PER_PACKET/8;
  long long int i1, i2, o1;
  
  // create streams
  cudaStream_t streams[2];
  for (int st=0;st<2;st++) 
    cudaStreamCreate(&streams[st]);

  // timing
  // copy, prepare, cublas, output
  clock_t begin, end;

  // do big memcpy
  //begin = clock();
  //cudaMemcpy(d->d_big_input,d->h_input,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4,cudaMemcpyHostToDevice);
  //end = clock();
  //d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // loop over halves of the array
  for (int iArm=0;iArm<2;iArm++) {
  
    // zero out output arrays
    cudaMemset(d->d_bigbeam_a_r,0,(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK*sizeof(half));
    cudaMemset(d->d_bigbeam_a_i,0,(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK*sizeof(half));
    cudaMemset(d->d_bigbeam_b_r,0,(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK*sizeof(half));
    cudaMemset(d->d_bigbeam_b_i,0,(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2*NPACKETS_PER_BLOCK*sizeof(half));
    cudaDeviceSynchronize();
    
    // copy data to device
    // initial data: [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
    // final data: need to split by NANTS.
    begin = clock();
    for (i1=0;i1<NPACKETS_PER_BLOCK;i1++) 
      cudaMemcpyAsync(d->d_input+i1*(NANTS/2)*NCHAN_PER_PACKET*4,d->h_input+i1*(NANTS)*NCHAN_PER_PACKET*4+iArm*(NANTS/2)*NCHAN_PER_PACKET*4,(NANTS/2)*NCHAN_PER_PACKET*4,cudaMemcpyHostToDevice,streams[iArm]);
    end = clock();
    d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // do reorder and fluff of data to real and imag
    begin = clock();
    dim3 dimBlock1(32, 8), dimGrid1(NCHAN_PER_PACKET*2/32,(NPACKETS_PER_BLOCK)*(NANTS/2)/32);
    transpose_fluff_bf<<<dimGrid1,dimBlock1,0,streams[iArm]>>>((unsigned short *)(d->d_input), d->d_bar, d->d_bai, d->d_bbr, d->d_bbi);
    end = clock();
    d->prep += (float)(end - begin) / CLOCKS_PER_SEC;

    // large matrix multiply to get real and imag outputs
    // set up for gemm
    cublasSetStream(cublasH, streams[iArm]);
    i2 = iArm*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2); // weights offset
    
    // run strided batched gemm
    begin = clock();

    // POL A
    
    // ac
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bar,lda,strideA,
			      d->weights_a_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_a_r,ldc,strideC,
			      batchCount);
    // -bd
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &malpha,d->d_bai,lda,strideA,
			      d->weights_a_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_a_r,ldc,strideC,
			      batchCount);
    // bc
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bai,lda,strideA,
			      d->weights_a_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_a_i,ldc,strideC,
			      batchCount);
    // ad
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bar,lda,strideA,
			      d->weights_a_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_a_i,ldc,strideC,
			      batchCount);

    // POL B
    
    // ac
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bbr,lda,strideA,
			      d->weights_b_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_b_r,ldc,strideC,
			      batchCount);
    // -bd
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &malpha,d->d_bbi,lda,strideA,
			      d->weights_b_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_b_r,ldc,strideC,
			      batchCount);
    // bc
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bbi,lda,strideA,
			      d->weights_b_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_b_i,ldc,strideC,
			      batchCount);
    // ad
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bbr,lda,strideA,
			      d->weights_b_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_b_i,ldc,strideC,
			      batchCount);

    cudaDeviceSynchronize();
    end = clock();
    d->cubl += (float)(end - begin) / CLOCKS_PER_SEC;
      
        
    // form total power, sum/transpose twice
    begin = clock();

    // incoherent beam summation
    sum_ib<<<NCHAN_PER_PACKET*2*NPACKETS_PER_BLOCK,32,0,streams[iArm]>>>(d->d_bar,d->d_bai,d->d_bbr,d->d_bbi,d->d_ibsum,d->d_flagants+iArm*48);
    
    dim3 dimBlock2(32, 8), dimGrid2(NPACKETS_PER_BLOCK/4/32,(NCHAN_PER_PACKET/8)*(NBEAMS/2)*8*2/32);
    power_sum_and_transpose_output<<<dimGrid2,dimBlock2,0,streams[iArm]>>>(d->d_bigbeam_a_r,d->d_bigbeam_b_r,d->d_bigbeam_a_i,d->d_bigbeam_b_i,d->d_ibsum,d->subtract_ib,d->d_htx);

    dim3 dimBlock(32, 8), dimGrid((NBEAMS/2)/32,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)/32);
    sum_transpose_and_scale_output<<<dimGrid,dimBlock,0,streams[iArm]>>>(d->d_htx,d->d_bigpower+iArm*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2),d->subtract_ib);

    end = clock();
    d->outp += (float)(end - begin) / CLOCKS_PER_SEC;
      

  }

  for (int st=0;st<2;st++) 
    cudaStreamDestroy(streams[st]);


  cublasDestroy(cublasH);

  // form sum over times
  sum_beam<<<NBEAMS,512>>>(d->d_bigpower,d->d_chscf);
  cudaMemcpy(d->h_chscf,d->d_chscf,4*NBEAMS,cudaMemcpyDeviceToHost);
  
}

// kernel to populate an instance of weights matrix [2, (NCHAN_PER_PACKET/8), NBEAMS/2, (NANTS/2)]
// run with 2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2)/128 blocks of 128 threads
__global__ void populate_weights_matrix(float * antpos_e, float * antpos_n, float * calibs, half * war, half * wai, half * wbr, half * wbi, float * fqs, float dec) {

  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int inidx = bidx*128+tidx;  
  
  // get indices
  int iArm = (int)(inidx / ((NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2)));
  int iidx = (int)(inidx % ((NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2)));
  int fq = (int)(iidx / ((NBEAMS/2)*(NANTS/2)));
  int idx = (int)(iidx % ((NBEAMS/2)*(NANTS/2)));
  int bm = (int)(idx / (NANTS/2));
  int a = (int)(idx % (NANTS/2));
  int widx = (a+48*iArm)*(NCHAN_PER_PACKET/8)*2*2 + fq*2*2;
  
  // calculate weights
  float theta, afac, twr, twi;
  if (iArm==0) {
    theta = sep*(127.-bm*1.)*PI/10800.; // radians
    afac = -2.*PI*fqs[fq]*theta/CVAC; // factor for rotate
    twr = cosf(afac*antpos_e[a+48*iArm]);
    twi = sinf(afac*antpos_e[a+48*iArm]);
    war[inidx] = __float2half((twr*calibs[widx] - twi*calibs[widx+1]));
    wai[inidx] = __float2half((twi*calibs[widx] + twr*calibs[widx+1]));
    wbr[inidx] = __float2half((twr*calibs[widx+2] - twi*calibs[widx+3]));
    wbi[inidx] = __float2half((twi*calibs[widx+2] + twr*calibs[widx+3]));
    //wr[inidx] = __float2half(calibs[widx]);
    //wi[inidx] = __float2half(calibs[widx+1]);
    //wr[inidx] = __float2half(1.0);
    //wi[inidx] = __float2half(0.0);
  }
  if (iArm==1) {
    theta = sep_ns*(127.-bm*1.)*PI/10800.-(PI/180.)*dec; // radians
    afac = -2.*PI*fqs[fq]*sinf(theta)/CVAC; // factor for rotate
    twr = cosf(afac*antpos_n[a+48*iArm]);
    twi = sinf(afac*antpos_n[a+48*iArm]);
    war[inidx] = __float2half((twr*calibs[widx] - twi*calibs[widx+1]));
    wai[inidx] = __float2half((twi*calibs[widx] + twr*calibs[widx+1]));
    wbr[inidx] = __float2half((twr*calibs[widx+2] - twi*calibs[widx+3]));
    wbi[inidx] = __float2half((twi*calibs[widx+2] + twr*calibs[widx+3]));
    //wr[inidx] = __float2half(calibs[widx]);
    //wi[inidx] = __float2half(calibs[widx+1]);
    //wr[inidx] = __float2half(1.0);
    //wi[inidx] = __float2half(0.0);
  }
    
}

// GPU-powered function to populate weights matrix for beamformer
// file format:
// sequential pairs of eastings and northings
// then [NANTS, 48, R/I] calibs

void calc_weights(dmem * d) {

  // allocate
  float *antpos_e = (float *)malloc(sizeof(float)*NANTS);
  float *antpos_n = (float *)malloc(sizeof(float)*NANTS);
  float *calibs = (float *)malloc(sizeof(float)*NANTS*(NCHAN_PER_PACKET/8)*2*2);
  float *d_antpos_e, *d_antpos_n, *d_calibs;
  int * flagas = (int *)malloc(sizeof(int)*NANTS);
  float wnorm;
  cudaMalloc((void **)(&d_antpos_e), sizeof(float)*NANTS);
  cudaMalloc((void **)(&d_antpos_n), sizeof(float)*NANTS);
  cudaMalloc((void **)(&d_calibs), sizeof(float)*NANTS*(NCHAN_PER_PACKET/8)*2*2);

  // deal with antpos and calibs
  int iant, found;
  for (int i=0;i<NANTS;i++) {
    antpos_e[i] = d->h_winp[i];
    antpos_n[i] = d->h_winp[i+NANTS];
  }
  for (int i=0;i<NANTS*(NCHAN_PER_PACKET/8)*2;i++) {

    iant = (int)(i/((NCHAN_PER_PACKET/8)*2));
    flagas[iant] = 0;

    found = 0;
    for (int j=0;j<d->nflags;j++) {
      if (d->flagants[j]==iant) {
	found = 1;
	flagas[iant] = 1;
      }
    }

    calibs[2*i] = d->h_winp[2*NANTS+2*i];
    calibs[2*i+1] = d->h_winp[2*NANTS+2*i+1];

    wnorm = sqrt(calibs[2*i]*calibs[2*i] + calibs[2*i+1]*calibs[2*i+1]);
    if (wnorm!=0.0) {
      calibs[2*i] /= wnorm;
      calibs[2*i+1] /= wnorm;
    }

    if (found==1) {
      calibs[2*i] = 0.;
      calibs[2*i+1] = 0.;
    }
  }

  //for (int i=0;i<NANTS*(NCHAN_PER_PACKET/8)*2;i++) printf("%f %f\n",calibs[2*i],calibs[2*i+1]);
  
  cudaMemcpy(d_antpos_e,antpos_e,NANTS*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_antpos_n,antpos_n,NANTS*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d->d_flagants,flagas,NANTS*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_calibs,calibs,NANTS*(NCHAN_PER_PACKET/8)*2*2*sizeof(float),cudaMemcpyHostToDevice);

  // run kernel to populate weights matrix
  //weights are [NCHAN_PER_PACKET/8, (NBEAMS/2), NANTS/2] for each pol and r/i, and arm
  populate_weights_matrix<<<2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*(NANTS/2)/128,128>>>(d_antpos_e,d_antpos_n,d_calibs,d->weights_a_r,d->weights_a_i,d->weights_b_r,d->weights_b_i,d->d_freqs,37.23-(d->obsdec));  
  
  // free stuff
  cudaFree(d_antpos_e);
  cudaFree(d_antpos_n);
  cudaFree(d_calibs);
  free(antpos_e);
  free(antpos_n);
  free(calibs);
  free(flagas);
  
}

// MAIN

int main (int argc, char *argv[]) {  

  cudaSetDevice(1);
  
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_bfCorr", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t in_key = REORDER_BLOCK_KEY;
  key_t out_key = XGPU_BLOCK_KEY;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  int bf = 0;
  int test = 0;
  float mydec = 71.66;
  char ftest[200], fflagants[200], fcalib[200], fpower[200];
  float sfreq = 1498.75;
  int subtract_ib = 0;
  
  while ((arg=getopt(argc,argv,"c:i:o:t:f:a:s:g:p:kbdh")) != -1)
    {
      switch (arg)
	{
	case 'c':
	  if (optarg)
	    {
	      core = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-c flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'i':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &in_key) != 1) {
		syslog(LOG_ERR, "could not parse key from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-i flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'o':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &out_key) != 1) {
		syslog(LOG_ERR, "could not parse key from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-o flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 't':
	  if (optarg)
            {
	      test = 1;
	      syslog(LOG_INFO, "test mode");
	      if (sscanf (optarg, "%s", &ftest) != 1) {
		syslog(LOG_ERR, "could not read test file name from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-t flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'a':
	  if (optarg)
            {
	      syslog(LOG_INFO, "read calib file %s",optarg);
	      if (sscanf (optarg, "%s", &fcalib) != 1) {
		syslog(LOG_ERR, "could not read calib file name from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-a flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'f':
	  if (optarg)
            {
	      syslog(LOG_INFO, "reading flag ants file %s",optarg);
	      if (sscanf (optarg, "%s", &fflagants) != 1) {
		syslog(LOG_ERR, "could not read flagants file name from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'p':
	  if (optarg)
            {
	      syslog(LOG_INFO, "writing power file %s",optarg);
	      if (sscanf (optarg, "%s", &fpower) != 1) {
		syslog(LOG_ERR, "could not read power file name from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-p flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 's':
	  if (optarg)
            {
	      sfreq = atof(optarg);
	      syslog(LOG_INFO, "start freq %g",sfreq);
 	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-s flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'g':
	  if (optarg)
            {
	      mydec = atof(optarg);
	      syslog(LOG_INFO, "obs dec %g",mydec);
 	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-g flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'b':
	  bf=1;
	  cudaSetDevice(0);
	  syslog (LOG_NOTICE, "Running beamformer, NOT correlator");
	  break;
	case 'k':
	  subtract_ib=1;
	  syslog (LOG_NOTICE, "Subtracting incoherent beam");
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // Bind to cpu core
  if (core >= 0)
    {
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
      syslog(LOG_NOTICE,"bound to core %d", core);
    }

  // allocate device memory
  dmem d;
  initialize(&d,bf,subtract_ib);

  // set up for beamformer
  FILE *ff, *fp;
  int iii;
  if (bf) {

    if (!(ff=fopen(fflagants,"r"))) {
      syslog(LOG_ERR,"could not open flagants file\n");
      exit(1);
    }
    d.nflags=0;
    iii = 0;
    while (!feof(ff)) {
      fscanf(ff,"%d\n",&d.flagants[iii]);
      d.nflags++;
      iii++;
    }
    fclose(ff);

    if (!(ff=fopen(fcalib,"rb"))) {
      syslog(LOG_ERR,"could not open calibss file\n");
      exit(1);
    }
    fread(d.h_winp,NANTS*2+NANTS*(NCHAN_PER_PACKET/8)*2*2,4,ff);
    fclose(ff);

    for (iii=0;iii<(NCHAN_PER_PACKET/8);iii++)
      d.h_freqs[iii] = 1e6*(sfreq-iii*250./1024.);
    cudaMemcpy(d.d_freqs,d.h_freqs,sizeof(float)*(NCHAN_PER_PACKET/8),cudaMemcpyHostToDevice);

    // calculate weights
    d.obsdec = mydec;
    calc_weights(&d);

    // open power
    if (!test)
      fp = fopen(fpower,"w");
    
  }

  // test mode
  FILE *fin, *fout;
  uint64_t sz, output_size, in_block_size, rd_size;
  in_block_size = NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2;
  char * output_data, * o1;
  int nreps = 1, nchunks = 1;
  if (test) {

    // read one block of input data

    // get size of file
    fin=fopen(ftest,"rb");
    fseek(fin,0L,SEEK_END);
    sz = ftell(fin);
    rewind(fin);

    // figure out how many reps and chunks to read with
    if (sz>in_block_size) {
      nreps = (int)(sz/in_block_size);
      rd_size = in_block_size;
    }
    else {
      nchunks = (int)(in_block_size/sz);
      rd_size =	sz;
    }

    // allocate input
    d.h_input = (char *)malloc(sizeof(char)*in_block_size);

    printf("NREPS NCHUNKS %d %d\n",nreps,nchunks);
    
    // loop over reps and chunks
    for (int reps=0; reps<nreps; reps++) {

      for (int chunks=0;chunks<nchunks;chunks++) {

	// read input file
	if (chunks>0) rewind(fin);
	fread(d.h_input+chunks*rd_size,rd_size,1,fin);

      }

      // run correlator or beamformer, and output data
      if (bf==0) {
	if (DEBUG) syslog(LOG_INFO,"run correlator");
	dcorrelator(&d);
	if (DEBUG) syslog(LOG_INFO,"copy to host");
	output_size = NBASE*NCHAN_PER_PACKET*2*2*4;
	output_data = (char *)malloc(output_size);
	cudaMemcpy(output_data,d.d_output,output_size,cudaMemcpyDeviceToHost);
	
	fout = fopen("output.dat","ab");
	fwrite((float *)output_data,sizeof(float),NBASE*NCHAN_PER_PACKET*2*2,fout);
	fclose(fout);
      }
      else {
	if (DEBUG) syslog(LOG_INFO,"run beamformer");
	dbeamformer(&d);
	syslog(LOG_INFO,"%f %f %f %f \n",d.cp,d.prep,d.cubl,d.outp);
	if (DEBUG) syslog(LOG_INFO,"copy to host");
	output_size = (NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*NBEAMS;
	output_data = (char *)malloc(output_size);
	cudaMemcpy(output_data,d.d_bigpower,output_size,cudaMemcpyDeviceToHost);	
	
	fout = fopen("output.dat","ab");
	fwrite((unsigned char *)output_data,sizeof(unsigned char),output_size,fout);
	fclose(fout);
      }

    }

	
    // free
    free(d.h_input);
    free(output_data);
    free(o1);
    deallocate(&d,bf);
    fclose(fin);
    exit(1);
  }
  


  
  // DADA stuff
  
  syslog (LOG_INFO, "creating in and out hdus");
  
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to dada buffer in");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to dada buffer in");
    return EXIT_FAILURE;
  }
  
  hdu_out  = dada_hdu_create ();
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog (LOG_ERR,"could not connect to output  buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write(hdu_out) < 0) {
    syslog (LOG_ERR, "could not lock to output buffer");
    return EXIT_FAILURE;
  }

  uint64_t header_size = 0;

  // deal with headers
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "could not read next header");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");  

  // register input with gpu
  dada_cuda_dbregister(hdu_in);
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %d %d\n",block_size,block_out);
  if (bf==0) 
    syslog(LOG_INFO, "main: EXPECT input and output block sizes %d %d\n",NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2,NBASE*NCHAN_PER_PACKET*2*2*4);
  else
    syslog(LOG_INFO, "main: EXPECT input and output block sizes %d %d\n",NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*NBEAMS);
  uint64_t  bytes_read = 0;
  char * block;
  char * output_buffer;
  output_buffer = (char *)malloc(block_out);
  uint64_t written, block_id;

  // output powers
  float output_power[NBEAMS];
  int iPower = 0;
  
  // get things started
  bool observation_complete=0;
  bool started = 0;
  syslog(LOG_INFO, "starting observation");
  int blocks = 0;
  clock_t begin, end;
  double time_spent;
  
  while (!observation_complete) {

    // zero out powers
    if (iPower==0) {
      for (int i=0;i<NBEAMS;i++) output_power[i] = 0.;
    }
    
    if (DEBUG) syslog(LOG_INFO,"reading block");    
    
    // open block
    d.h_input = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    // do stuff
    //begin = clock();
    if (bf==0) {
      if (DEBUG) syslog(LOG_INFO,"run correlator");
      dcorrelator(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      cudaMemcpy(output_buffer,d.d_output,block_out,cudaMemcpyDeviceToHost);
    }
    else {
      if (DEBUG) syslog(LOG_INFO,"run beamformer");
      dbeamformer(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      cudaMemcpy(output_buffer,d.d_bigpower,block_out,cudaMemcpyDeviceToHost);            
      
      // deal with power output
      for (int i=0;i<NBEAMS;i++)
	output_power[i] += d.h_chscf[i]/(1.*power_cycle);
	//fprintf(fp,"%g\n",d.h_chscf[i]);

      iPower++;
      if (iPower == power_cycle) {
	for (int i=0;i<NBEAMS;i++)
	  fprintf(fp,"%g\n",output_power[i]);
	iPower = 0;
      }
      
    }
    //end = clock();
    //time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    //cout << "spent time " << d.cp << " " << d.prep << " " << d.cubl << " " << d.outp << " s" << endl;
    
    // write to output
    
    written = ipcio_write (hdu_out->data_block, (char *)(output_buffer), block_out);
    if (written < block_out)
      {
	syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	return EXIT_FAILURE;
      }
    
    if (DEBUG) syslog(LOG_INFO, "written block %d",blocks);	    
    blocks++;

    
      
    // finish up
    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    
  }

  // finish up
  free(output_buffer);
  deallocate(&d,bf);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


