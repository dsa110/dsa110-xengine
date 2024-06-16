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

// Initialise device memory
void initialize(dmem * d, int bf);

// Deallocate device memory
void deallocate(dmem * d, int bf);

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);

// ?
int dada_bind_thread_to_core(int core);
