#pragma once

template <typename prec> void promoteComplexCharToFloat(prec *output, const char *input, const int rows, const int cols);
template <typename prec> void host_MdagM_gemm(const prec *A, const prec *B, prec *C, const int m, const int n, const int k);
template <typename prec> prec test_hermiticity(const prec *C, const int m, const int n);
