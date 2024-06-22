#pragma once
#include "dsaX.h"
#include "dsaX_cuda_headers.h"

void dsaXHgemmStridedBatchedCuda(void *real_in, void *imag_in, void *real_out, void *imag_out, dsaXBLASParam param);
