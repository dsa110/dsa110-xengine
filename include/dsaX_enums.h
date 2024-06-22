#pragma once

#define DSA_INVALID_ENUM (-0x7fffffff - 1)

typedef enum dsaError_t { DSA_SUCCESS = 0, DSA_ERROR = 1, DSA_ERROR_UNINITIALIZED = 2 } dsaError_t;

typedef enum dsaBLASOperation_s {				 
  DSA_BLAS_OP_N = 0, // No transpose
  DSA_BLAS_OP_T = 1, // Transpose only
  DSA_BLAS_OP_C = 2, // Conjugate transpose
  DSA_BLAS_OP_INVALID = DSA_INVALID_ENUM
} dsaBLASOperation;

typedef enum dsaXBLASType_s {
  DSA_BLAS_GEMM = 0,
  DSA_BLAS_INVALID = DSA_INVALID_ENUM
} dsaXBLASType;

typedef enum dsaXBLASDataType_s {
  DSA_BLAS_DATATYPE_H = 0, // Half
  DSA_BLAS_DATATYPE_S = 1, // Single
  DSA_BLAS_DATATYPE_D = 2, // Double
  DSA_BLAS_DATATYPE_HC = 3, // Complex(half)
  DSA_BLAS_DATATYPE_C = 4, // Complex(single)
  DSA_BLAS_DATATYPE_Z = 5, // Complex(double)
  DSA_BLAS_DATATYPE_INVALID = DSA_INVALID_ENUM
} dsaXBLASDataType;

typedef enum dsaXBLASDataOrder_s {
  DSA_BLAS_DATAORDER_ROW = 0,
  DSA_BLAS_DATAORDER_COL = 1,
  DSA_BLAS_DATAORDER_INVALID = DSA_INVALID_ENUM
} dsaXBLASDataOrder;
