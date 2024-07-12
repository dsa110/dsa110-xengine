#include "utils.h"

/**
 * Promote complex char riri... data to planar half rr.. ii.. 
 *
 * @param[out] inr float precision real array
 * @param[out] ini float precision imag array
 * @param[in]  input char precision complex array
 * @param[in]  rows number of rows
 * @param[in]  cols number of cols
 */
template <typename prec> void promoteComplexCharToFloat(prec *output, const char *input, const int rows, const int cols) {
  
#pragma omp parallel for collapse(2)
  int idx = 0;
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      int idx = i * rows + j;
      
      // 15 in unsigned char binary is 00001111. Perform bitwise & on 15 and input char data iiiirrrr
      // to get real part 4 bit data.
      // 0000rrrr
      // Bit shift this result by 4 to the left.
      // rrrr0000
      // Cast to signed char.
      // +-rrr0000
      // Bitshift mantisa only to the right by 4 bits
      // +-0000rrr
      // Cast to float and use CUDA intrinsic to cast to signed half
      output[2*idx] = (prec)((char)((   (unsigned char)(input[2*idx]) & (unsigned char)(15)  ) << 4) >> 4);
      
      // 240 in unsigned char binary is 11110000. Perform bitwise & on 240 and input char data iiiirrrr
      // to get imag part 4 bit data
      // iiii0000.
      // Cast to signed char
      // +-iii0000
      // Bitshift mantisa only to the right by 4 bits
      // +-0000iii
      // Cast to float and use CUDA intrinsic to cast to signed half
      output[2*idx+1] = (prec)((char)((   (unsigned char)(input[2*idx+1]) & (unsigned char)(240)  )) >> 4);
    }
  }
}

// Assume ROW ordered data in interleaved format
template <typename prec> void host_MdagM_gemm(const prec *A, const prec *B, prec *C, const int m, const int n, const int k) {
  
#pragma omp parallel for collapse(2)
  for(int i=0; i<m; i++) {
    for(int j=0; j<n; j++) {
      
      // Get C index
      int C_idx = i * n + j;
      C[2*C_idx]   = 0.0;
      C[2*C_idx+1] = 0.0;
      for(int l=0; l<k; l++) {
	
	int A_idx = l + m + i;
	int B_idx = l * n + j;

	// Compute Adag * B = C
	C[2*C_idx]   += A[2*A_idx] * B[2*B_idx] + A[2*A_idx+1] * B[2*B_idx+1];
	C[2*C_idx+1] += A[2*A_idx] * B[2*B_idx+1] - A[2*A_idx+1] * B[2*B_idx];
      }
    }
  }
}

// Assume ROW ordered data in interleaved format
template <typename prec> prec test_hermiticity(const prec *C, const int m, const int n) {

  prec frob_norm = 0.0;
  
#pragma omp parallel for collapse(2) reduction (+:frob_norm)
  for(int i=0; i<m; i++) {
    for(int j=0; j<n; j++) {
      
      int C_idx  = i + n * j;
      int Cd_idx = j + m * i;

      double diff = pow((C[2*C_idx] - C[2*Cd_idx]), 2);
      diff       += pow((C[2*C_idx+1] - C[2*Cd_idx+1]), 2);
      frob_norm = frob_norm + diff;
      
 
    }
  }
  return frob_norm;
}
