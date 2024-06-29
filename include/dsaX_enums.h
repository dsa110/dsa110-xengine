#pragma once

#define DSA_INVALID_ENUM (-0x7fffffff - 1)

typedef enum dsaXError_t {
  DSA_SUCCESS = 0,
  DSA_ERROR = 1,
  DSA_ERROR_UNINITIALIZED = 2,
  DSA_ERROR_INVALID = DSA_INVALID_ENUM
} dsaXError;

typedef enum dsaXBLASOperation_s {				 
  DSA_BLAS_OP_N = 0, // No transpose
  DSA_BLAS_OP_T = 1, // Transpose only
  DSA_BLAS_OP_A = 2, // Adjoint imaginary, no transpose
  DSA_BLAS_OP_C = 3, // Conjugate transpose
  DSA_BLAS_OP_INVALID = DSA_INVALID_ENUM
} dsaXBLASOperation;

typedef enum dsaXBLASType_s {
  DSA_BLAS_GEMM = 0,
  DSA_BLAS_INVALID = DSA_INVALID_ENUM
} dsaXBLASType;

typedef enum dsaXBLASLib_s {
  DSA_BLAS_LIB_CUBLAS = 0,
  DSA_BLAS_LIB_MAGMA  = 1,
  DSA_BLAS_LIB_CUTLASS = 2,
  DSA_BLAS_LIB_TCC = 3, 
  DSA_BLAS_LIB_OPENBLAS = 4,
  DSA_BLAS_LIB_NATIVE = 5, 
  DSA_BLAS_LIB_INVALID = DSA_INVALID_ENUM  
} dsaXBLASLib;

typedef enum dsaXBLASDataType_s {				
  DSA_BLAS_DATATYPE_H = 0, // Half
  DSA_BLAS_DATATYPE_S = 1, // Single
  DSA_BLAS_DATATYPE_D = 2, // Double
  DSA_BLAS_DATATYPE_HC = 3, // Complex(half)
  DSA_BLAS_DATATYPE_C = 4, // Complex(single)
  DSA_BLAS_DATATYPE_Z = 5, // Complex(double)
  DSA_BLAS_DATATYPE_4b_REAL = 6, // 4b sized real
  DSA_BLAS_DATATYPE_2b_REAL = 7, // 2b sized real
  DSA_BLAS_DATATYPE_4b_COMPLEX = 8, // Char sized complex (4b,4b)
  DSA_BLAS_DATATYPE_2b_COMPLEX = 9, // 4b sized (2b,2b)  
  DSA_BLAS_DATATYPE_INVALID = DSA_INVALID_ENUM
} dsaXBLASDataType;

typedef enum dsaXBLASDataOrder_s {
  DSA_BLAS_DATAORDER_ROW = 0,
  DSA_BLAS_DATAORDER_COL = 1,
  DSA_BLAS_DATAORDER_INVALID = DSA_INVALID_ENUM
} dsaXBLASDataOrder;

typedef enum dsaXMemcpyKind_s {
  dsaXMemcpyHostToHost = 0,
  dsaXMemcpyHostToDevice = 1,
  dsaXMemcpyDeviceToHost = 2,
  dsaXMemcpyDeviceToDevice = 3,
  dsaXMemcpyInvalid = DSA_INVALID_ENUM
} dsaXMemcpyKind;
