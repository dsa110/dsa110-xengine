#pragma once 

#include <iostream>
#include <algorithm>
#include <complex>
#include <vector>
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

#include "dsaX_cuda_headers.h"
#include "dsaX_psrdada_headers.h"

// required to prevent overflow in corr matrix multiply
#define halfFac 4

// beam sep
#define sep 1.0 // arcmin

/* global variables */
//#define DEBUG;

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

// Structure that carries BLAS parameters
typedef struct dsaBLASParam_s {  
  size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and DSA see the same struct*/
  
  dsaBLASType blas_type;    /**< Type of BLAS computation to perfrom */
  
  // GEMM params
  dsaBLASOperation trans_a; /**< operation op(A) that is non- or (conj.) transpose. */
  dsaBLASOperation trans_b; /**< operation op(B) that is non- or (conj.) transpose. */
  int m;                     /**< number of rows of matrix op(A) and C. */
  int n;                     /**< number of columns of matrix op(B) and C. */
  int k;                     /**< number of columns of op(A) and rows of op(B). */
  int lda;                   /**< leading dimension of two-dimensional array used to store the matrix A. */
  int ldb;                   /**< leading dimension of two-dimensional array used to store matrix B. */
  int ldc;                   /**< leading dimension of two-dimensional array used to store matrix C. */
  int a_offset;              /**< position of the A array from which begin read/write. */
  int b_offset;              /**< position of the B array from which begin read/write. */
  int c_offset;              /**< position of the C array from which begin read/write. */
  int a_stride;              /**< stride of the A array in strided(batched) mode */
  int b_stride;              /**< stride of the B array in strided(batched) mode */
  int c_stride;              /**< stride of the C array in strided(batched) mode */
  std::complex<double> alpha;             /**< scalar used for multiplication. */
  std::complex<double>  beta;             /**< scalar used for multiplication. If beta==0, C does not have to be a valid input. */
  
  // Common params
  int batch_count;             /**< number of pointers contained in arrayA, arrayB and arrayC. */
  dsaBLASDataType data_type;   /**< Specifies if using S(C) or D(Z) BLAS type */
  dsaBLASDataOrder data_order; /**< Specifies if using Row or Column major */
  
} dsaBLASParam;
  

// Initialise device memory
void initialize_device_memeory(dmem * d, int bf);

// Deallocate device memory
void deallocate(dmem * d, int bf);

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);

// ?
int dada_bind_thread_to_core(int core);
