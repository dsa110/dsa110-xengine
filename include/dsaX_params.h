#pragma once

#include <complex>

#include "dsaX_enums.h"

// Structure that carries BLAS parameters
// This should be able to communicate to all
// backend choices of BLAS library
typedef struct dsaXBLASParam_s {  
  size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and DSA see the same struct*/
  
  dsaXBLASType blas_type;    /**< Type of BLAS computation to perform */

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

// Structure that carries Correlator class parameters
typedef struct dsaXCorrParam_s {  
  size_t struct_size;        /**< Size of this struct in bytes.  Used to ensure that the host application and DSA see the same struct*/
  
  dsaXBLASLib blas_lib;         /**< Which BLAS library to use for BLAS ops */
  dsaXBLASDataType data_type;   /**< Specifies if using S(C) or D(Z) BLAS type */
  dsaXBLASDataOrder data_order; /**< Specifies if using Row or Column major */
  
} dsaXCorrParam;

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

// Parameter struct helper functions for user
const char *getBLASLibString(dsaXBLASLib lib);
const char *getBLASDataTypeString(dsaXBLASDataType type);
const char *getBLASDataOrderString(dsaXBLASDataOrder order);
void printDsaXBLASParam(const dsaXBLASParam param);
void printDsaXCorrParam(const dsaXCorrParam param);

// Create params
dsaXCorrParam newDsaXCorrParam(void);
