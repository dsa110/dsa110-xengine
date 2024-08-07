// -*- c++ -*-

#include <time.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <dedisp.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <src/sigproc.h>

#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"

using namespace std;

#define WIDTH 600000
#define HEIGHT 768

int main() {

    float * h_arr = (float *)malloc(sizeof(float)*WIDTH*HEIGHT);
    float * d_arr;
    int stride;
    cudaMallocPitch((void **)(&d_arr), (size_t *)(&stride), (unsigned long)(WIDTH*sizeof(float)), HEIGHT);
    
    for (int i=0;i<HEIGHT;i++) {
      for (int j=0;j<WIDTH;j++)
	h_arr[i*WIDTH+j] = i*WIDTH+j

}
