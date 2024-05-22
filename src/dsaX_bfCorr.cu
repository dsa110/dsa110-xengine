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

// beam sep
#define sep 1.0 // arcmin

/* global variables */
int DEBUG = 0;

// define structure that carries around device memory
typedef struct dmem {

  // initial data and streams
  char * h_input; // host input pointer
  char * d_input, * d_tx; // [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
  
  // correlator pointers
  // giant array for r and i: [NCHAN_PER_PACKET, 2 pol, NANTS_PROCESS, NPACKETS_PER_BLOCK * 2 times]
  half * d_r, * d_i;
  // arrays for matrix multiply output: input [NANTS_PROCESS, NANTS_PROCESS]
  half * d_outr, *d_outi, *d_tx_outr, *d_tx_outi;
  // giant output array: [NBASE, NCHAN_PER_PACKET, 2 pol, 2 complex]
  float * d_output;
  
  // beamformer pointers
  char * d_big_input;
  half * d_br, * d_bi;
  half * weights_r, * weights_i; //weights: [arm, tactp, b]
  half * d_bigbeam_r, * d_bigbeam_i; //output: [tc, b]
  unsigned char * d_bigpower; //output: [b, tc]
  float * d_scf; // scale factor per beam
  float * d_chscf;
  float * h_winp;
  int * flagants, nflags;
  float * h_freqs, * d_freqs;

  // timing
  float cp, prep, cubl, outp;
  
} dmem;


// allocate device memory
void initialize(dmem * d, int bf) {
  
  // for correlator
  if (bf==0) {
    cudaMalloc((void **)(&d->d_input), sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_r), sizeof(half)*NCHAN_PER_PACKET*2*NANTS*NPACKETS_PER_BLOCK*2);
    cudaMalloc((void **)(&d->d_i), sizeof(half)*NCHAN_PER_PACKET*2*NANTS*NPACKETS_PER_BLOCK*2);
    cudaMalloc((void **)(&d->d_tx), sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_output), sizeof(float)*NBASE*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_outr), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
    cudaMalloc((void **)(&d->d_outi), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
    cudaMalloc((void **)(&d->d_tx_outr), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
    cudaMalloc((void **)(&d->d_tx_outi), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
  }

  // for beamformer
  if (bf==1) {
    cudaMalloc((void **)(&d->d_input), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS/2)*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_big_input), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS)*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_tx), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS/2)*NCHAN_PER_PACKET*2*2);
    cudaMalloc((void **)(&d->d_br), sizeof(half)*NCHAN_PER_PACKET*2*(NANTS/2)*(NPACKETS_PER_BLOCK)*2);
    cudaMalloc((void **)(&d->d_bi), sizeof(half)*NCHAN_PER_PACKET*2*(NANTS/2)*(NPACKETS_PER_BLOCK)*2);
    cudaMalloc((void **)(&d->weights_r), sizeof(half)*2*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8));
    cudaMalloc((void **)(&d->weights_i), sizeof(half)*2*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8));
    cudaMalloc((void **)(&d->d_bigbeam_r), sizeof(half)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2));
    cudaMalloc((void **)(&d->d_bigbeam_i), sizeof(half)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2));
    cudaMalloc((void **)(&d->d_bigpower), sizeof(unsigned char)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS));
    cudaMalloc((void **)(&d->d_scf), sizeof(float)*(NBEAMS/2)); // beam scale factor
    cudaMalloc((void **)(&d->d_chscf), sizeof(float)*(NBEAMS/2)*(NCHAN_PER_PACKET/8)); // beam scale factor

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
  }
  if (bf==1) {
    cudaFree(d->d_tx);
    cudaFree(d->d_br);
    cudaFree(d->d_bi);
    cudaFree(d->weights_r);
    cudaFree(d->weights_i);
    cudaFree(d->d_bigbeam_r);
    cudaFree(d->d_bigbeam_i);
    cudaFree(d->d_bigpower);
    cudaFree(d->d_scf);
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
	 " -s start frequency (assumes -0.244140625MHz BW)\n");
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

  // zero out output arrays
  cudaMemset(d->d_outr,0,NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(half));
  cudaMemset(d->d_outi,0,NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(half));
  cudaMemset(d->d_output,0,NCHAN_PER_PACKET*2*NANTS*NANTS*sizeof(float));
  
  // copy to device
  cudaMemcpy(d->d_input,d->h_input,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2,cudaMemcpyHostToDevice);

  // reorder input
  reorder_input(d->d_input,d->d_tx,d->d_r,d->d_i);

  // not sure if essential
  cudaDeviceSynchronize();
  
  // set up for gemm
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

  // destroy stream
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  
  // reorder output data
  reorder_output(d);
  
}

// kernels to reorder and fluff input data for beamformer
// initial data is [NPACKETS_PER_BLOCK, (NANTS/2), NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]            
// want [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, (NANTS/2), 8chan, 2 times, 2 pol, 4-bit complex]      // run as 16x16 tiled transpose with 32-byte words 
// launch with dim3 dimBlock(16, 8) and dim3 dimGrid(Width/16, Height/16)
// here, width=NCHAN_PER_PACKET/8 is the dimension of the fastest input index
// dim3 dimBlock1(16, 8), dimGrid1(NCHAN_PER_PACKET/8/16, (NPACKETS_PER_BLOCK)*(NANTS/2)/16);
__global__ void transpose_input_bf(double * idata, double * odata) {

  __shared__ double tile[16][17][4];
  
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  int width = gridDim.x * 16;

  for (int j = 0; j < 16; j += 8) {
    tile[threadIdx.y+j][threadIdx.x][0] = idata[4*((y+j)*width + x)];
    tile[threadIdx.y+j][threadIdx.x][1] = idata[4*((y+j)*width + x)+1];
    tile[threadIdx.y+j][threadIdx.x][2] = idata[4*((y+j)*width + x)+2];
    tile[threadIdx.y+j][threadIdx.x][3] = idata[4*((y+j)*width + x)+3];
  }
  
  __syncthreads();

  x = blockIdx.y * 16 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 16 + threadIdx.y;
  width = gridDim.y * 16;

  for (int j = 0; j < 16; j += 8) {
    odata[4*((y+j)*width + x)] = tile[threadIdx.x][threadIdx.y + j][0];
    odata[4*((y+j)*width + x)+1] = tile[threadIdx.x][threadIdx.y + j][1];
    odata[4*((y+j)*width + x)+2] = tile[threadIdx.x][threadIdx.y + j][2];
    odata[4*((y+j)*width + x)+3] = tile[threadIdx.x][threadIdx.y + j][3];
  }

}

// kernel to fluff input bf data
// run with NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128 blocks of 128 threads
__global__ void fluff_input_bf(char * input, half * dr, half * di) {

  int bidx = blockIdx.x; // assume NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128
  int tidx = threadIdx.x; // assume 128
  int idx = bidx*128+tidx;

  dr[idx] = __float2half(0.015625*((float)((char)(((unsigned char)(input[idx]) & (unsigned char)(15)) << 4) >> 4)));
  di[idx] = __float2half(0.015625*((float)((char)(((unsigned char)(input[idx]) & (unsigned char)(240))) >> 4)));
  
}

// transpose, add and scale kernel for bf
// assume breakdown into tiles of 16x16, and run with 16x8 threads per block
// launch with dim3 dimBlock(16, 8) and dim3 dimGrid((NBEAMS/2)*(NPACKETS_PER_BLOCK/4)/16, (NCHAN_PER_PACKET/8)/16)
// scf is a per-beam scale factor to enable recasting as unsigned char
__global__ void transpose_scale_bf(half * ir, half * ii, unsigned char * odata) {

  __shared__ float tile[16][17];
  
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  int width = gridDim.x * 16;
  float dr, di;

  for (int j = 0; j < 16; j += 8) {
    dr = (float)(ir[(y+j)*width + x]);
    di = (float)(ii[(y+j)*width + x]);
    tile[threadIdx.y+j][threadIdx.x] = (dr*dr+di*di);
  }

  __syncthreads();

  x = blockIdx.y * 16 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 16 + threadIdx.y;
  width = gridDim.y * 16;

  for (int j = 0; j < 16; j += 8)
    odata[(y+j)*width + x] = (unsigned char)(tile[threadIdx.x][threadIdx.y + j]/128.);

}

// sum over all times in output beam array
// run with (NCHAN_PER_PACKET/8)*(NBEAMS/2) blocks of (NPACKETS_PER_BLOCK/4) threads
__global__ void sum_beam(unsigned char * input, float * output) {

  __shared__ float summ[512];
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int idx = bidx*256+tidx;
  int bm = (int)(bidx/48);
  int ch = (int)(bidx % 48);

  summ[tidx] = (float)(input[bm*256*48 + tidx*48 + ch]);

  __syncthreads();

  if (tidx<256) {
    summ[tidx] += summ[tidx+256];
    summ[tidx] += summ[tidx+128];
    summ[tidx] += summ[tidx+64];
    summ[tidx] += summ[tidx+32];
    summ[tidx] += summ[tidx+16];
    summ[tidx] += summ[tidx+8];
    summ[tidx] += summ[tidx+4];
    summ[tidx] += summ[tidx+2];
    summ[tidx] += summ[tidx+1];
  }

  if (tidx==0) output[bidx] = summ[tidx];
  
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
void dbeamformer(dmem * d) {

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
  long long int i1, i2, o1;
  
  // create streams
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // timing
  // copy, prepare, cublas, output
  clock_t begin, end;

  // do big memcpy
  begin = clock();
  cudaMemcpy(d->d_big_input,d->h_input,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4,cudaMemcpyHostToDevice);
  end = clock();
  d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // loop over halves of the array
  for (int iArm=0;iArm<2;iArm++) {
  
    // zero out output arrays
    cudaMemset(d->d_bigbeam_r,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(half));
    cudaMemset(d->d_bigbeam_i,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(half));
    cudaDeviceSynchronize();
    
    // copy data to device
    // initial data: [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
    // final data: need to split by NANTS.
    begin = clock();
    for (i1=0;i1<NPACKETS_PER_BLOCK;i1++) 
      cudaMemcpy(d->d_input+i1*(NANTS/2)*NCHAN_PER_PACKET*4,d->d_big_input+i1*(NANTS)*NCHAN_PER_PACKET*4+iArm*(NANTS/2)*NCHAN_PER_PACKET*4,(NANTS/2)*NCHAN_PER_PACKET*4,cudaMemcpyDeviceToDevice);
    end = clock();
    d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // do reorder and fluff of data to real and imag
    begin = clock();
    dim3 dimBlock1(16, 8), dimGrid1(NCHAN_PER_PACKET/8/16, (NPACKETS_PER_BLOCK)*(NANTS/2)/16);
    transpose_input_bf<<<dimGrid1,dimBlock1>>>((double *)(d->d_input),(double *)(d->d_tx));
    fluff_input_bf<<<NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128,128>>>(d->d_tx,d->d_br,d->d_bi);
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

// kernel to populate an instance of weights matrix [2, (NCHAN_PER_PACKET/8), NBEAMS/2, 4times*(NANTS/2)*8chan*2tim*2pol]
// run with 2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)/128 blocks of 128 threads
__global__ void populate_weights_matrix(float * antpos_e, float * antpos_n, float * calibs, half * wr, half * wi, float * fqs) {

  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int inidx = bidx*128+tidx;  
  
  // 2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)
  
  // get indices
  int iArm = (int)(inidx / ((NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)));
  int iidx = (int)(inidx % ((NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)));
  int fq = (int)(iidx / (128*(NANTS/2)*(NBEAMS/2)));
  int idx = (int)(iidx % (128*(NANTS/2)*(NBEAMS/2)));
  int bm = (int)(idx / (128*(NANTS/2)));
  int tactp = (int)(idx % (128*(NANTS/2)));
  int t = (int)(tactp / (32*(NANTS/2)));
  int actp = (int)(tactp % (32*(NANTS/2)));
  int a = (int)(actp / 32);
  int ctp = (int)(actp % 32);
  int c = (int)(ctp / 4);
  int tp = (int)(ctp % 4);
  int t2 = (int)(tp / 2);
  int pol = (int)(tp % 2);
  int widx = (a+48*iArm)*(NCHAN_PER_PACKET/8)*2*2 + fq*2*2 + pol*2;
  
  // calculate weights
  float theta, afac, twr, twi;
  if (iArm==0) {
    theta = sep*(127.-bm*1.)*PI/10800.; // radians
    afac = -2.*PI*fqs[fq]*theta/CVAC; // factor for rotate
    twr = cos(afac*antpos_e[a+48*iArm]);
    twi = sin(afac*antpos_e[a+48*iArm]);
    wr[inidx] = __float2half((twr*calibs[widx] - twi*calibs[widx+1]));
    wi[inidx] = __float2half((twi*calibs[widx] + twr*calibs[widx+1]));
    //wr[inidx] = __float2half(calibs[widx]);
    //wi[inidx] = __float2half(calibs[widx+1]);
  }
  if (iArm==1) {
    theta = sep*(127.-bm*1.)*PI/10800.; // radians
    afac = -2.*PI*fqs[fq]*theta/CVAC; // factor for rotate
    twr = cos(afac*antpos_n[a+48*iArm]);
    twi = sin(afac*antpos_n[a+48*iArm]);
    wr[inidx] = __float2half((twr*calibs[widx] - twi*calibs[widx+1]));
    wi[inidx] = __float2half((twi*calibs[widx] + twr*calibs[widx+1]));
    //wr[inidx] = __float2half(calibs[widx]);
    //wi[inidx] = __float2half(calibs[widx+1]);
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
  float wnorm;
  cudaMalloc((void **)(&d_antpos_e), sizeof(float)*NANTS);
  cudaMalloc((void **)(&d_antpos_n), sizeof(float)*NANTS);
  cudaMalloc((void **)(&d_calibs), sizeof(float)*NANTS*(NCHAN_PER_PACKET/8)*2*2);

  // deal with antpos and calibs
  int iant, found;
  for (int i=0;i<NANTS;i++) {
    antpos_e[i] = d->h_winp[2*i];
    antpos_n[i] = d->h_winp[2*i+1];
  }
  for (int i=0;i<NANTS*(NCHAN_PER_PACKET/8)*2;i++) {

    iant = (int)(i/((NCHAN_PER_PACKET/8)*2));

    found = 0;
    for (int j=0;j<d->nflags;j++)
      if (d->flagants[j]==iant) found = 1;

    calibs[2*i] = d->h_winp[2*NANTS+2*i];
    calibs[2*i+1] = d->h_winp[2*NANTS+2*i+1];

    wnorm = sqrt(calibs[2*i]*calibs[2*i] + calibs[2*i+1]*calibs[2*i+1]);
    if (wnorm!=0.0) {
      calibs[2*i] /= wnorm;
      calibs[2*i+1] /= wnorm;
    }

    //if (found==1) {
    //calibs[2*i] = 0.;
    //calibs[2*i+1] = 0.;
    //}
  }

  //for (int i=0;i<NANTS*(NCHAN_PER_PACKET/8)*2;i++) printf("%f %f\n",calibs[2*i],calibs[2*i+1]);
  
  cudaMemcpy(d_antpos_e,antpos_e,NANTS*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_antpos_n,antpos_n,NANTS*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_calibs,calibs,NANTS*(NCHAN_PER_PACKET/8)*2*2*sizeof(float),cudaMemcpyHostToDevice);

  // run kernel to populate weights matrix
  populate_weights_matrix<<<2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)/128,128>>>(d_antpos_e,d_antpos_n,d_calibs,d->weights_r,d->weights_i,d->d_freqs);  
  
  // free stuff
  cudaFree(d_antpos_e);
  cudaFree(d_antpos_n);
  cudaFree(d_calibs);
  free(antpos_e);
  free(antpos_n);
  free(calibs);
  
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
  char ftest[200], fflagants[200], fcalib[200];
  float sfreq = 1498.75;

  
  while ((arg=getopt(argc,argv,"c:i:o:t:f:a:s:bdh")) != -1)
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
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'b':
	  bf=1;
	  syslog (LOG_NOTICE, "Running beamformer, NOT correlator");
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
  initialize(&d,bf);

  // set up for beamformer
  FILE *ff;
  int iii;
  if (bf) {

    if (!(ff=fopen(fflagants,"r"))) {
      syslog(LOG_ERR,"could not open flagants file\n");
      exit(1);
    }
    d.nflags=0;
    while (!feof(ff)) {
      fscanf(ff,"%d\n",&d.flagants[iii]);
      d.nflags++;
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
    calc_weights(&d);
    
  }

  // test mode
  FILE *fin, *fout;
  uint64_t output_size;
  char * output_data, * o1;
  if (test) {

    // read one block of input data    
    d.h_input = (char *)malloc(sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
    for (int i=0;i<512;i++) {
      fin = fopen(ftest,"rb");
      fread(d.h_input+i*4*NANTS*NCHAN_PER_PACKET*2*2,4*NANTS*NCHAN_PER_PACKET*2*2,1,fin);
      fclose(fin);
    }

    // run correlator or beamformer, and output data
    if (bf==0) {
      if (DEBUG) syslog(LOG_INFO,"run correlator");
      dcorrelator(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      output_size = NBASE*NCHAN_PER_PACKET*2*2*4;
      output_data = (char *)malloc(output_size);
      cudaMemcpy(output_data,d.d_output,output_size,cudaMemcpyDeviceToHost);

      fout = fopen("output.dat","wb");
      fwrite((float *)output_data,sizeof(float),NBASE*NCHAN_PER_PACKET*2*2,fout);
      fclose(fout);
    }
    else {
      if (DEBUG) syslog(LOG_INFO,"run beamformer");
      dbeamformer(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      output_size = (NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*NBEAMS;
      output_data = (char *)malloc(output_size);
      cudaMemcpy(output_data,d.d_bigpower,output_size,cudaMemcpyDeviceToHost);

      /*output_size = 2*2*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8);
      o1 = (char *)malloc(output_size);
      cudaMemcpy(o1,d.weights_r,output_size,cudaMemcpyDeviceToHost);*/
	
      

      fout = fopen("output.dat","wb");
      fwrite((unsigned char *)output_data,sizeof(unsigned char),output_size,fout);
      //fwrite(o1,1,output_size,fout);
      fclose(fout);
    }

	
    // free
    free(d.h_input);
    free(output_data);
    free(o1);
    deallocate(&d,bf);

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
  
  // get things started
  bool observation_complete=0;
  bool started = 0;
  syslog(LOG_INFO, "starting observation");
  int blocks = 0;
  clock_t begin, end;
  double time_spent;
  
  while (!observation_complete) {

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
    }
    //end = clock();
    //time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << "spent time " << d.cp << " " << d.prep << " " << d.cubl << " " << d.outp << " s" << endl;
    
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


