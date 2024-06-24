#pragma once

#include "dsaX.h"

void dsaXmemset(void *array, int ch, size_t n);

void dsaXmemcpyHostToDevice(void *array_device, void *array_host, size_t n);
void dsaXmemcpyDeviceToHost(void *array_host, void *array_device, size_t n);
void dsaXmemcpyDeviceToDevice(void *array_device_to, void *array_device_from, size_t n);
