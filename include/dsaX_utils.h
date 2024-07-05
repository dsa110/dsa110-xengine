#pragma once

#include "dsaX_params.h"
#include "timer.h"

void dsaXmemset(void *array, int ch, size_t n);

void dsaXmemcpy(void *array_out, void *array_in, size_t n, dsaXMemcpyKind kind, int stream = 0);

void dsaXDeviceSynchronize();
