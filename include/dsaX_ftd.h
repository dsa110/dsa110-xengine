#pragma once

//#include "dsaX_def.h"
#include "dsaX_enums.h"
#include "dsaX_params.h"

// define structures that carry around memory pointers
// and metric.
// DMH: make a base and inherit into corr and bf
typedef struct dmem_corr_s {
  
  // initial data and streams
  char *h_input;        // host input pointer
  char *d_input, *d_tx; // [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
  
  // correlator pointers
  // giant array for r and i: [NCHAN_PER_PACKET, 2 pol, NANTS_PROCESS, NPACKETS_PER_BLOCK *2 times]
  void *d_r, *d_i; //half
  // arrays for matrix multiply output: input [NANTS_PROCESS, NANTS_PROCESS]
  void *d_outr, *d_outi, *d_tx_outr, *d_tx_outi; //half
  // giant output array: [NBASE, NCHAN_PER_PACKET, 2 pol, 2 complex]
  float *d_output;

  metrics metric_data;
  
} dmem_corr;

typedef struct dmem_bf_s {

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
  metrics metric_data;
  
} dmem_bf;

void dcorrelator(dmem_corr *d);

class Correlator {
  
private:
protected:
  
  dmem_corr d;  
  dsaXCorrParam corr_param;
  dsaXBLASParam blas_param;
  
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

void destroyDsaXCorrDeviceMemory(dmem_corr *d);
void initDsaXCorrDeviceMemory(dmem_corr *d);

void reorderCorrelatorOutput(dmem_corr *d);
void reorderCorrelatorInput(dmem_corr *d);
