#pragma once

#include <vector>

#include "utils.h"

#ifdef DSA_XENGINE_TARGET_CUDA
#include "cuda_headers.h"

static std::vector<cudaStream_t> streams;
static cublasHandle_t cublasH = NULL;

static bool cublas_init = false;
static bool stream_init = false;

cudaStream_t get_stream(unsigned int i);
#endif

void init_streams(unsigned int n_streams);
void destroy_streams();
