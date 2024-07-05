// -*- c++ -*-
/* assumes input and output block size is appropriate - will seg fault otherwise*/
/*
Workflow is similar for BF and corr applications
 - copy data to GPU, convert to half-precision and calibrate while reordering
 - do matrix operations to populate large output vector
 */

#include <iostream>
#include <cstring>

#include "dsaX_def.h"
#include "dsaX.h"
#include "fast_time_domain.h"
#include "blas_interface.h"
#include "utils.h"
#include "psrdada_utils.h"

using namespace std;

Correlator::Correlator(const dsaXCorrParam *param) {

  // Transfer passed param to internal objects
  corr_param = *param;
  d.corr_param = *param;

  // Select back end BLAS engine 
  blas_param.struct_size = sizeof(blas_param);
  blas_param.blas_type = DSA_BLAS_GEMM;
  blas_param.blas_lib = corr_param.blas_lib;

  // Streams will be class specific
  // so launch and destroy in the class
  initStreams(corr_param.n_streams);
  
  // Initialise device memeory
  d.dev_malloc_timer.start();
  initDsaXCorrDeviceMemory(&d, corr_param.n_streams);
  d.dev_malloc_timer.stop();

  // Compute indices
  computeIndices(&d);
  
  // gemm settings
  // input: [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
  // output: [NCHAN_PER_PACKET, 2times, 2pol, NANTS, NANTS]
#if defined OLD_BLAS
  //cout << "Old params" << endl;  
  blas_param.data_order = DSA_BLAS_DATAORDER_COL;
  blas_param.trans_a = DSA_BLAS_OP_A;
  blas_param.trans_b = DSA_BLAS_OP_T;
  blas_param.m = NANTS;
  blas_param.n = NANTS;
  blas_param.k = NPACKETS_PER_BLOCK/halfFac;
  blas_param.alpha = 1.0;
  blas_param.lda = blas_param.m;
  blas_param.ldb = blas_param.n;
  blas_param.beta = 0.;
  blas_param.ldc = blas_param.m;
  blas_param.a_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;
  blas_param.b_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;
  blas_param.c_stride = NANTS*NANTS;
  blas_param.batch_count = NCHAN_PER_PACKET*2*2*halfFac;
  blas_param.a_offset = 0;
  blas_param.b_offset = 0;
  blas_param.c_offset = 0;
#else
  //cout << "My params" << endl;
  blas_param.data_order = DSA_BLAS_DATAORDER_ROW;
  blas_param.trans_a = DSA_BLAS_OP_C;
  blas_param.trans_b = DSA_BLAS_OP_N;
  blas_param.m = NANTS;
  blas_param.n = NANTS;
  blas_param.k = NPACKETS_PER_BLOCK/halfFac;
  blas_param.alpha = 1.0;
  blas_param.lda = blas_param.m;
  blas_param.ldb = blas_param.n;
  blas_param.beta = 0.;
  blas_param.ldc = blas_param.m;
  blas_param.a_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;;
  blas_param.b_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;;
  blas_param.c_stride = NANTS*NANTS;
  blas_param.batch_count = NCHAN_PER_PACKET*2*2*halfFac;
  blas_param.a_offset = 0;
  blas_param.b_offset = 0;
  blas_param.c_offset = 0;
#endif
  
  // Swap A and B if in row order
  if (blas_param.data_order == DSA_BLAS_DATAORDER_ROW) {
    swap(blas_param.m, blas_param.n);
    swap(blas_param.lda, blas_param.ldb);
    swap(blas_param.trans_a, blas_param.trans_b);
    swap(blas_param.a_offset, blas_param.b_offset);
    swap(blas_param.a_stride, blas_param.b_stride);
    //swap(A_data, B_data);
    //swap(A_data, B_data);
  }

  printDsaXBLASParam(blas_param);
  
  flops = 8; // 8 complex flops per element
  flops *= blas_param.m;
  flops *= blas_param.n;
  flops *= blas_param.k;
  flops *= blas_param.batch_count;
  
  cout << "Correlator flops = 2*M*N*K * batch = (" << 2 << "*"<< blas_param.m << "*" << blas_param.n << "*" << blas_param.k << "*" << blas_param.batch_count << ") = " << flops << endl;
  cout << "Correlator Gflop = " << (1e-9)*flops << endl;

  // DMH: reset counters method
  
}

Correlator::~Correlator() {

  // Clean up memory
  destroyDsaXCorrDeviceMemory(&d);
  destroyStreams();
  
  // Transfer metrics to 
  double device_malloc_time = (1.0*d.dev_malloc_timer.elapsed().count())/(1e6);
  double host_malloc_time = (1.0*d.host_malloc_timer.elapsed().count())/(1e6);
  double device_compute_time = (1.0*d.dev_compute_timer.elapsed().count())/(1e6);
  cout << "Correlator malloc time device  = " << device_malloc_time << " seconds." << endl;
  cout << "Correlator malloc time host    = " << host_malloc_time << " seconds." << endl;  
  cout << "Correlator compute time device = " << device_compute_time << " seconds. " << endl;
  
  double h2d_time = (1.0*d.H2D_timer.elapsed().count())/(1e6);
  cout << "Correlator H2D time            = " << h2d_time << " seconds. ";
  cout << "Bandwidth " << (1.0*d.H2D_bytes)/pow(1024,3) / h2d_time << " Gbytes/second." << endl;
  
  double d2h_time = (1.0*d.D2H_timer.elapsed().count())/(1e6);
  cout << "Correlator D2H time            = " << d2h_time << " seconds. ";
  cout << "Bandwidth " << (1.0*d.D2H_bytes)/pow(1024,3) / d2h_time << " Gbytes/second." << endl;

  double h2h_time = (1.0*d.H2H_timer.elapsed().count())/(1e6);
  cout << "Correlator H2H time            = " << h2h_time << " seconds. ";
  cout << "Bandwidth " << (1.0*d.H2H_bytes)/pow(1024,3) / h2h_time << " Gbytes/second." << endl;  

  double total = device_malloc_time + host_malloc_time + device_compute_time + h2d_time + d2h_time;
  cout << "Correlator TOTAL time          = " << total << " seconds. " << endl;
  
  double Tflops = (1.0*d.dev_compute_timer.iterations()*(1e-12*flops)/device_compute_time);
  cout << "Correlator Tflops              = " << Tflops <<  endl;
}

void Correlator::compute(void *output, void *input) {
  
  uint64_t in_stream_block = sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2;
  uint64_t out_stream_block = sizeof(float)*NBASE*NCHAN_PER_PACKET*2*2;

  unsigned int n_streams = corr_param.n_streams;
  
  // Ensure output array is zero
  dsaXmemset(d.d_output, 0, n_streams * out_stream_block);
  
  // Loop over the array in streams for concurrency.
  for(int i=0; i<n_streams; i++) {
    // copy to device  
    dsaXmemcpy(d.d_input + i*in_stream_block, (char*)input + i*in_stream_block, in_stream_block, dsaXMemcpyHostToDeviceAsync, i);
      
    // reorder input into real and imaginary planar complex
    // arrays and, if required, promote to required precision
    // for consumption by BLAS engine.
    promoteComplexCharToPlanarHalf(&d, i);
    //reorderCorrInput(&d, i);
    
    // Perform GEMM accoring to back end configuration
    dsaXHgemmStridedBatched((short*)d.d_r + i*in_stream_block, (short*)d.d_i + i*in_stream_block,
			    (short*)d.d_r + i*in_stream_block, (short*)d.d_i + i*in_stream_block,
			    (short*)d.d_outr + i*in_stream_block, (short*)d.d_outi + i*in_stream_block, blas_param, i);
    
    // Reorder output data back to interleaved complex
    // and promote to float
    reorderCorrOutput(&d, i);
    
    // Pass result back to host
    d.D2H_timer.start();
    dsaXmemcpy((float*)output + i*out_stream_block, d.d_output + i*out_stream_block, out_stream_block, dsaXMemcpyDeviceToHostAsync, i);

    d.D2H_bytes += out_stream_block;
    d.D2H_timer.stop();
  }

  // End loop over stream. Sync to device prior to handing back
  // scope to client program.
  dsaXDeviceSynchronize();
}

 
// correlator function
// workflow: copy to device, reorder, stridedBatchedGemm, reorder, copy back to host
// DMH: CUDA references excised. Make me a class
void dcorrelator(corr_handle *d) {

  // zero out output arrays
  dsaXmemset(d->d_outr, 0, NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(short)); //half -> short
  dsaXmemset(d->d_outi, 0, NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(short)); //half -> short
  dsaXmemset(d->d_output, 0, NCHAN_PER_PACKET*2*NANTS*NANTS*sizeof(float));

  // copy to device
  dsaXmemcpy(d->d_input, d->h_input, NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2, dsaXMemcpyHostToDevice);
  
  // reorder input into real and imaginary arrays of 2 byte data
  reorderCorrInput(d, 0);
  
  dsaXBLASParam blas_param;
  blas_param.struct_size = sizeof(blas_param);
  blas_param.blas_type = DSA_BLAS_GEMM;

  // gemm settings
  // input: [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
  // output: [NCHAN_PER_PACKET, 2times, 2pol, NANTS, NANTS]

#if defined OLD_BLAS
  //cout << "Old params" << endl;
  
  blas_param.data_order = DSA_BLAS_DATAORDER_COL;
  blas_param.trans_a = DSA_BLAS_OP_A;
  blas_param.trans_b = DSA_BLAS_OP_T;
  blas_param.m = NANTS;
  blas_param.n = NANTS;
  blas_param.k = NPACKETS_PER_BLOCK/halfFac;
  blas_param.alpha = 1.0;
  blas_param.lda = blas_param.m;
  blas_param.ldb = blas_param.n;
  blas_param.beta = 0.;
  blas_param.ldc = blas_param.m;
  blas_param.a_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;
  blas_param.b_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;
  blas_param.c_stride = NANTS*NANTS;
  blas_param.batch_count = NCHAN_PER_PACKET*2*2*halfFac;
  blas_param.a_offset = 0;
  blas_param.b_offset = 0;
  blas_param.c_offset = 0;
#else
  //cout << "My params" << endl;
  
  blas_param.data_order = DSA_BLAS_DATAORDER_ROW;
  blas_param.trans_a = DSA_BLAS_OP_C;
  blas_param.trans_b = DSA_BLAS_OP_N;
  blas_param.m = NANTS;
  blas_param.n = NANTS;
  blas_param.k = NPACKETS_PER_BLOCK/halfFac;
  blas_param.alpha = 1.0;
  blas_param.lda = blas_param.m;
  blas_param.ldb = blas_param.n;
  blas_param.beta = 0.;
  blas_param.ldc = blas_param.m;
  blas_param.a_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;;
  blas_param.b_stride = NPACKETS_PER_BLOCK*NANTS/halfFac;;
  blas_param.c_stride = NANTS*NANTS;
  blas_param.batch_count = NCHAN_PER_PACKET*2*2*halfFac;
  blas_param.a_offset = 0;
  blas_param.b_offset = 0;
  blas_param.c_offset = 0;
#endif

  // Swap A and B if in row order
  if (blas_param.data_order == DSA_BLAS_DATAORDER_ROW) {
    swap(blas_param.m, blas_param.n);
    swap(blas_param.lda, blas_param.ldb);
    swap(blas_param.trans_a, blas_param.trans_b);
    swap(blas_param.a_offset, blas_param.b_offset);
    swap(blas_param.a_stride, blas_param.b_stride);
    //swap(A_data, B_data);
    //swap(A_data, B_data);
  }  

  
  //printDsaXBLASParam(blas_param);
  
  // DMH: fix me
  blas_param.blas_lib = DSA_BLAS_LIB_CUBLAS;
  
  // Perform GEMM accoring to back end configuration
  dsaXHgemmStridedBatched(d->d_r, d->d_i, d->d_r, d->d_i, d->d_outr, d->d_outi, blas_param);

  //for(int i=0; i<8; i++) inspectPackedData(d.h_input[i], i);
  
  // reorder output data
  reorderCorrOutput(d);
}
