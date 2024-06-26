#pragma once

#include "dsaX.h"

void dsaXmemset(void *array, int ch, size_t n);
void dsaXmemcpy(void *array_out, void *array_in, size_t n, dsaXMemcpyKind kind);
void dsaXDeviceSynchronize();
