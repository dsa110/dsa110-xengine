#pragma once 

#include <complex>

#include "dsaX_def.h"
#include "dsaX_enums.h"

#define OLD_BLAS

// Structure that carries BLAS parameters
typedef struct dsaXBLASParam_s {  
  size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and DSA see the same struct*/
  
  dsaXBLASType blas_type;    /**< Type of BLAS computation to perfrom */

  dsaXBLASLib blas_lib;      /**< Which BLAS library to use for BLAS ops */
  
  // GEMM params
  dsaXBLASOperation trans_a; /**< operation op(A) that is non- or (conj.) transpose. */
  dsaXBLASOperation trans_b; /**< operation op(B) that is non- or (conj.) transpose. */
  int m;                     /**< number of rows of matrix op(A) and C. */
  int n;                     /**< number of columns of matrix op(B) and C. */
  int k;                     /**< number of columns of op(A) and rows of op(B). */
  int lda;                   /**< leading dimension of two-dimensional array used to store the matrix A. */
  int ldb;                   /**< leading dimension of two-dimensional array used to store matrix B. */
  int ldc;                   /**< leading dimension of two-dimensional array used to store matrix C. */
  long long int a_offset;    /**< position of the A array from which begin read/write. */
  long long int b_offset;    /**< position of the B array from which begin read/write. */
  long long int c_offset;    /**< position of the C array from which begin read/write. */
  long long int a_stride;    /**< stride of the A array in strided(batched) mode */
  long long int b_stride;    /**< stride of the B array in strided(batched) mode */
  long long int c_stride;    /**< stride of the C array in strided(batched) mode */
  std::complex<double> alpha;     /**< scalar used for multiplication. */
  std::complex<double>  beta;     /**< scalar used for multiplication. If beta==0, C does not have to be a valid input. */
  
  // Common params
  int batch_count;             /**< number of pointers contained in arrayA, arrayB and arrayC. */
  dsaXBLASDataType data_type;   /**< Specifies if using S(C) or D(Z) BLAS type */
  dsaXBLASDataOrder data_order; /**< Specifies if using Row or Column major */
  
} dsaXBLASParam;

// Structure that carries BLAS parameters
typedef struct dsaXCorrParam_s {  
  size_t struct_size;        /**< Size of this struct in bytes.  Used to ensure that the host application and DSA see the same struct*/
  
  dsaXBLASLib blas_lib;         /**< Which BLAS library to use for BLAS ops */
  dsaXBLASDataType data_type;   /**< Specifies if using S(C) or D(Z) BLAS type */
  dsaXBLASDataOrder data_order; /**< Specifies if using Row or Column major */
  
} dsaXCorrParam;

void printDsaXBLASParam(const dsaXBLASParam param);

// required to prevent overflow in corr matrix multiply
#define halfFac 4

// beam sep
#define sep 1.0 // arcmin

// Global timing and metrics structure for dsaX 
typedef struct metrics_s {

  // Mem copy times
  double mem_copy_time_H2H;
  double mem_copy_time_H2D;
  double mem_copy_time_D2H;
  double mem_copy_time_D2D;

  // Mem copy size
  double mem_copy_size_H2H;
  double mem_copy_size_H2D;
  double mem_copy_size_D2H;
  double mem_copy_size_D2D;

  // Compute
  double compute_time;
  double compute_flops;

  // Initialisation
  double initialisation_time;
} metrics;
  
// define structure that carries around memory pointers
// and timer for the correlator
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

  // timing
  float cp, prep, cubl, outp;
  
} dmem_bf;



void dsaXInit(int device_ordinal = 0);

void inspectPackedData(char input, int i, bool non_zero = false);

void dsaXCorrelator(void *output_data, void *input_data);

void reorderCorrelatorOutput(dmem_corr *d);
void reorderCorrelatorInput(dmem_corr *d);
