#pragma once

#include "dsaX_enums.h"
#include "dsaX_params.h"
#include "timer.h"

using ms = std::chrono::microseconds;
using hrc = std::chrono::high_resolution_clock;

// define structures that carry around memory pointers
// and metric.
// DMH: make a base and inherit into corr and bf
typedef struct corr_handle_s {
  
  // initial data and streams
  char *h_input;        // host input pointer
  char *d_input, *d_tx; // [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
  
  // DMH: fix me
  void *d_idxs;
  
  // correlator pointers
  // giant array for r and i: [NCHAN_PER_PACKET, 2 pol, NANTS_PROCESS, NPACKETS_PER_BLOCK *2 times]
  void *d_r, *d_i; //half
  // arrays for matrix multiply output: input [NANTS_PROCESS, NANTS_PROCESS]
  void *d_outr, *d_outi, *d_tx_outr, *d_tx_outi; //half
  // giant output array: [NBASE, NCHAN_PER_PACKET, 2 pol, 2 complex]
  float *d_output;

  dsaXCorrParam corr_param;

  double device_compute_flops;
  double host_compute_flops;
  
  double H2D_bytes;
  double D2H_bytes;
  double D2D_bytes;
  double H2H_bytes;

  // See 'using' at top of file for ms, hrc
  timer::Timer<ms, hrc> dev_compute_timer;
  timer::Timer<ms, hrc> dev_malloc_timer;
  timer::Timer<ms, hrc> dev_memset_timer;
  
  timer::Timer<ms, hrc> H2D_timer;
  timer::Timer<ms, hrc> D2H_timer;
  timer::Timer<ms, hrc> D2D_timer;
  timer::Timer<ms, hrc> H2H_timer;
  
  timer::Timer<ms, hrc> host_compute_timer;  
  timer::Timer<ms, hrc> host_malloc_timer;
  timer::Timer<ms, hrc> host_memset_timer;
  timer::Timer<ms, hrc> host_copy_timer;
  
} corr_handle;

typedef struct bf_handle_s {

  // beamformer pointers
  char *h_input;        // host input pointer
  char *d_input, *d_tx; // [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
  char *d_big_input;
  void *d_br, *d_bi; //half
  void *weights_r, *weights_i; //weights: [arm, tactp, b] //half
  void *d_bigbeam_r, *d_bigbeam_i; //output: [tc, b] //half
  unsigned char *d_bigpower; //output: [b, tc]
  float *d_scf; // scale factor per beam
  float *d_chscf;
  float *h_winp;
  int *flagants, nflags;
  float *h_freqs, *d_freqs;

  // timing (old)
  float cp, prep, cubl, outp;

  // See 'using' at top of file ms, hrc
  timer::Timer<ms, hrc> dev_compute_timer;
  timer::Timer<ms, hrc> dev_malloc_timer;
  timer::Timer<ms, hrc> dev_memset_timer;
    
  timer::Timer<ms, hrc> H2D_timer;
  timer::Timer<ms, hrc> D2H_timer;
  
  timer::Timer<ms, hrc> host_compute_timer;  
  timer::Timer<ms, hrc> host_malloc_timer;
  timer::Timer<ms, hrc> host_memset_timer;
  timer::Timer<ms, hrc> host_copy_timer;
  
} bf_handle;

// Deprecated function, remove after development
void dcorrelator(corr_handle *d);

// Base class
class dsaXBase {
  
 private:
 protected:

 public:
  dsaXBase();  
  ~dsaXBase();
  
};

class Correlator : public dsaXBase {
  
private:
protected:

  corr_handle d;  
  dsaXCorrParam corr_param;
  dsaXBLASParam blas_param;

  uint64_t flops;
  
public:
  
  // Constructor
  // Initialise device memory if CUDA enabled
  // make host memory if CPU
  Correlator(const dsaXCorrParam *corr_param);

  // Compute the FX correlator on input,
  // place result in output.
  void compute(void *output, void *input);
  
  ~Correlator();  
};


void initDsaXCorrDeviceMemory(corr_handle *d, unsigned int n_streams);
void destroyDsaXCorrDeviceMemory(corr_handle *d);
void promoteComplexCharToPlanarHalf(corr_handle *d, unsigned int n_streams);

void initBLAS();
void destroyBLAS();

void initStreams(unsigned int n);
void destroyStreams();

void computeIndices(corr_handle *d);
void reorderCorrelatorOutput(corr_handle *d, int stream);
void reorderCorrelatorInput(corr_handle *d, int stream);

