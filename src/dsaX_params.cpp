#include <iostream>

#include "dsaX_params.h"

using namespace std;

const char *getBLASLibString(dsaXBLASLib lib)
{
  const char *ret;

  switch (lib) {
  case DSA_BLAS_LIB_CUBLAS: ret = "CUBLAS"; break;
  case DSA_BLAS_LIB_MAGMA: ret = "MAGMA"; break;
  case DSA_BLAS_LIB_CUTLASS: ret = "CUTLAS"; break;
  case DSA_BLAS_LIB_OPENBLAS: ret = "OPENBLAS"; break;
  case DSA_BLAS_LIB_NATIVE: ret = "NATIVE"; break;
  default: ret = "unknown"; break;
  }
  
  return ret;
}

const char *getBLASDataTypeString(dsaXBLASDataType type)
{
  const char *ret;

  switch (type) {
  case DSA_BLAS_DATATYPE_H: ret = "Half"; break;
  case DSA_BLAS_DATATYPE_S: ret = "Single"; break;
  case DSA_BLAS_DATATYPE_D: ret = "Double"; break;
  case DSA_BLAS_DATATYPE_HC: ret = "Complex(half)"; break;
  case DSA_BLAS_DATATYPE_C: ret = "Complex(single)"; break;
  case DSA_BLAS_DATATYPE_Z: ret = "Complex(double)"; break;
  case DSA_BLAS_DATATYPE_4b_REAL: ret = "4b sized real"; break;
  case DSA_BLAS_DATATYPE_2b_REAL: ret = "2b sized real"; break;
  case DSA_BLAS_DATATYPE_4b_COMPLEX: ret = "Char sized complex (4b,4b)"; break;
  case DSA_BLAS_DATATYPE_2b_COMPLEX: ret = "4b sized (2b,2b)"; break;  
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *getBLASDataOrderString(dsaXBLASDataOrder order)
{
  const char *ret;

  switch (order) {
  case DSA_BLAS_DATAORDER_ROW: ret = "Row order"; break;
  case DSA_BLAS_DATAORDER_COL: ret = "Column order"; break;
  default: ret = "unknown"; break;
  }
  
  return ret;
}

void printDsaXCorrParam(const dsaXCorrParam param) {

  cout << "--- dsaXCorrParam begin ---" << endl;
  cout << "struct_size = " << param.struct_size << endl;
  cout << "blas_lib = " << getBLASLibString(param.blas_lib) << endl;
  cout << "data_type = " << getBLASDataTypeString(param.data_type) << endl;
  cout << "data_order = " << getBLASDataOrderString(param.data_order) << endl;
  cout << " --- dsaXCorrParam end ---" << endl;
}

void printDsaXBLASParam(const dsaXBLASParam param) {

  cout << " --- dsaXBLASParam begin ---" << endl;
  cout << "struct_size = " << param.struct_size << endl;
  cout << "blas_type = " << param.blas_type << endl;
  cout << "blas_lib = " << param.blas_lib << endl;
  cout << "data_type = " << param.data_type << endl;
  cout << "data_order = " << param.data_order << endl;
  cout << "trans_a = " << param.trans_a << endl;
  cout << "trans_b = " << param.trans_b << endl;
  cout << "m = " << param.m << endl;
  cout << "n = " << param.n << endl;
  cout << "k = " << param.k << endl;
  cout << "lda = " << param.lda << endl;
  cout << "ldb = " << param.ldb << endl;
  cout << "ldc = " << param.ldc << endl;
  cout << "a_offset = " << param.a_offset << endl;
  cout << "b_offset = " << param.b_offset << endl;
  cout << "c_offset = " << param.c_offset << endl;
  cout << "a_stride = " << param.a_stride << endl;
  cout << "b_stride = " << param.b_stride << endl;
  cout << "c_stride = " << param.c_stride << endl;
  cout << "alpha = " << param.alpha << endl;
  cout << "beta = " << param.beta << endl;
  cout << "batch_count = " << param.batch_count << endl;
  cout << " --- dsaXBLASParam end ---" << endl;
}

dsaXCorrParam newDsaXCorrParam(void) {
  dsaXCorrParam new_param;
  new_param.struct_size = sizeof(new_param);
  new_param.blas_lib = DSA_BLAS_LIB_INVALID;
  new_param.data_type = DSA_BLAS_DATATYPE_INVALID;
  new_param.data_order = DSA_BLAS_DATAORDER_INVALID;
  return new_param;
}
