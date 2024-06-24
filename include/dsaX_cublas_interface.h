#pragma once
#include "dsaX.h"
#include "dsaX_cuda_headers.h"

void dsaXHgemmStridedBatchedCuda(half *real_in, half *imag_in, half *real_out, half *imag_out, dsaXBLASParam param);
