#pragma once 

// Expose the use to compile time definitions,
// enums, parameters, and classes
#include "dsaX_def.h"
#include "enums.h"
#include "params.h"
#include "fast_time_domain.h"

// Use manual transpose route
// Uncomment to try new pure cuBLAS
//#define OLD_BLAS

// required to prevent overflow in corr matrix multiply
#define halfFac 4

// beam sep
#define sep 1.0 // arcmin

void dsaXInit(int device_ordinal = -1);

void dsaXEnd();



void *dsaXHostRegister(size_t size);

void inspectPackedData(char input, int i, bool non_zero = false);

void dsaXCorrelator(void *output_data, void *input_data, dsaXCorrParam *param);
