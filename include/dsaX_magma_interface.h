#pragma once
#include "dsaX.h"

void dsaXHgemmStridedBatchedMagma(void *real_a, void *imag_a, void *real_b, void *imag_b, void *real_c, void *imag_c, dsaXBLASParam param);
