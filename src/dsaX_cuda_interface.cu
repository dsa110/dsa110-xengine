#include <iostream>
#include <vector>

#include "dsaX_cuda_headers.h"
#include "dsaX_cuda_interface.h"
#include "dsaX_cuda_kernels.h"

using namespace std;

void dsaXInitCuda(int dev){
  cudaSetDevice(dev);
}

// allocate device memory
void initializeCorrCudaMemory(dmem_corr *d) {
  
  // for correlator
  cudaMalloc((void **)(&d->d_input), sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  cudaMalloc((void **)(&d->d_r), sizeof(half)*NCHAN_PER_PACKET*2*NANTS*NPACKETS_PER_BLOCK*2);
  cudaMalloc((void **)(&d->d_i), sizeof(half)*NCHAN_PER_PACKET*2*NANTS*NPACKETS_PER_BLOCK*2);
  cudaMalloc((void **)(&d->d_tx), sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
  cudaMalloc((void **)(&d->d_output), sizeof(float)*NBASE*NCHAN_PER_PACKET*2*2);
  cudaMalloc((void **)(&d->d_outr), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
  cudaMalloc((void **)(&d->d_outi), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
  cudaMalloc((void **)(&d->d_tx_outr), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
  cudaMalloc((void **)(&d->d_tx_outi), sizeof(half)*NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac);
}

void initializeBFCudaMemory(dmem_bf *d) {
  
  // for beamformer
  cudaMalloc((void **)(&d->d_input), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS/2)*NCHAN_PER_PACKET*2*2);
  cudaMalloc((void **)(&d->d_big_input), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS)*NCHAN_PER_PACKET*2*2);
  cudaMalloc((void **)(&d->d_tx), sizeof(char)*(NPACKETS_PER_BLOCK)*(NANTS/2)*NCHAN_PER_PACKET*2*2);
  cudaMalloc((void **)(&d->d_br), sizeof(half)*NCHAN_PER_PACKET*2*(NANTS/2)*(NPACKETS_PER_BLOCK)*2);
  cudaMalloc((void **)(&d->d_bi), sizeof(half)*NCHAN_PER_PACKET*2*(NANTS/2)*(NPACKETS_PER_BLOCK)*2);
  cudaMalloc((void **)(&d->weights_r), sizeof(half)*2*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8));
  cudaMalloc((void **)(&d->weights_i), sizeof(half)*2*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8));
  cudaMalloc((void **)(&d->d_bigbeam_r), sizeof(half)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2));
  cudaMalloc((void **)(&d->d_bigbeam_i), sizeof(half)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2));
  cudaMalloc((void **)(&d->d_bigpower), sizeof(unsigned char)*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS));
  cudaMalloc((void **)(&d->d_scf), sizeof(float)*(NBEAMS/2)); // beam scale factor
  cudaMalloc((void **)(&d->d_chscf), sizeof(float)*(NBEAMS/2)*(NCHAN_PER_PACKET/8)); // beam scale factor
  
  // input weights: first is [NANTS, E/N], then [NANTS, 48, 2pol, R/I]
  d->h_winp = (float *)malloc(sizeof(float)*(NANTS*2+NANTS*(NCHAN_PER_PACKET/8)*2*2));
  d->flagants = (int *)malloc(sizeof(int)*NANTS);
  d->h_freqs = (float *)malloc(sizeof(float)*(NCHAN_PER_PACKET/8));
  cudaMalloc((void **)(&d->d_freqs), sizeof(float)*(NCHAN_PER_PACKET/8));
  
  // timers
  d->cp = 0.;
  d->prep = 0.;
  d->outp = 0.;
  d->cubl = 0.;
}

// deallocate device memory
void deallocateCorrCudaMemory(dmem_corr *d) {
  
  cudaFree(d->d_input);
  cudaFree(d->d_r);
  cudaFree(d->d_i);
  cudaFree(d->d_tx);
  cudaFree(d->d_output);
  cudaFree(d->d_outr);
  cudaFree(d->d_outi);
  cudaFree(d->d_tx_outr);
  cudaFree(d->d_tx_outi);
}

// deallocate device memory
void deallocateBFCudaMemory(dmem_bf *d) {

  cudaFree(d->d_input);
  cudaFree(d->d_tx);
  cudaFree(d->d_br);
  cudaFree(d->d_bi);
  cudaFree(d->weights_r);
  cudaFree(d->weights_i);
  cudaFree(d->d_bigbeam_r);
  cudaFree(d->d_bigbeam_i);
  cudaFree(d->d_bigpower);
  cudaFree(d->d_scf);
  cudaFree(d->d_chscf);
  free(d->h_winp);
  free(d->flagants);
  cudaFree(d->d_freqs);
  free(d->h_freqs);
}  


// function to copy d_outr and d_outi to d_output
// inputs are [NCHAN_PER_PACKET, 2 time, 2 pol, NANTS, NANTS]
// the corr matrices are column major order
// output needs to be [NBASE, NCHAN_PER_PACKET, 2 pol, 2 complex]
// start with transpose to get [NANTS*NANTS, NCHAN_PER_PACKET*2*2], then sum into output using kernel
void reorderCorrOutputCuda(dmem_corr * d) {
  
  // transpose input data
#if defined (OLD_BLAS)
  dim3 dimBlock(32, 8), dimGrid((NANTS*NANTS)/32, (NCHAN_PER_PACKET*2*2*halfFac)/32);
  transpose_matrix<<<dimGrid, dimBlock>>>((half*)d->d_outr, (half*)d->d_tx_outr);
  transpose_matrix<<<dimGrid, dimBlock>>>((half*)d->d_outi, (half*)d->d_tx_outi);
#endif  
  // look at output
  /*char * odata = (char *)malloc(sizeof(char)*384*4*NANTS*NANTS*2*halfFac);
  cudaMemcpy(odata,d->d_tx_outr,384*4*NANTS*NANTS*2*halfFac,cudaMemcpyDeviceToHost);
  FILE *fout;
  fout=fopen("test2.test","wb");
  fwrite(odata,sizeof(char),384*4*NANTS*NANTS*2*halfFac,fout);
  fclose(fout);*/

  
  /*
  // set up for geam
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublasH, stream);

  // transpose output matrices into tx_outr and tx_outi
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = NCHAN_PER_PACKET*2*2;
  const int n = NANTS*NANTS/16; // columns in output
  const double alpha = 1.0;
  const double beta = 0.0;
  const int lda = n;
  const int ldb = m;
  const int ldc = ldb;
  cublasDgeam(cublasH,transa,transb,m,n,
	      &alpha,(double *)(d->d_outr),
	      lda,&beta,(double *)(d->d_tx_outr),
	      ldb,(double *)(d->d_tx_outr),ldc);
  cublasDgeam(cublasH,transa,transb,m,n,
	      &alpha,(double *)(d->d_outi),
	      lda,&beta,(double *)(d->d_tx_outi),
	      ldb,(double *)(d->d_tx_outi),ldc);
  */
  // now run kernel to sum into output
  int * h_idxs = (int *)malloc(sizeof(int)*NBASE);
  int * d_idxs;
  cudaMalloc((void **)(&d_idxs), sizeof(int)*NBASE);
  int ii = 0;
  // upper triangular order (column major) to match xGPU (not the same as CASA!)
  for (int i=0;i<NANTS;i++) {
    for (int j=0;j<=i;j++) {
      h_idxs[ii] = i*NANTS + j;
      ii++;
    }
  }
  cudaMemcpy(d_idxs, h_idxs, sizeof(int)*NBASE,cudaMemcpyHostToDevice);

  // run kernel to finish things
  // TUNABLE
  int blockDim = 128;
  int blocks = NCHAN_PER_PACKET*2*NBASE/blockDim;
#if defined (OLD_BLAS)
  corr_output_copy<<<blocks, blockDim>>>((half*)d->d_tx_outr, (half*)d->d_tx_outi, d->d_output, d_idxs);
#else
  corr_output_copy<<<blocks, blockDim>>>((half*)d->d_outr, (half*)d->d_outi, d->d_output, d_idxs);
#endif
  
  /*char * odata = (char *)malloc(sizeof(char)*384*4*NBASE*4);
  cudaMemcpy(odata,d->d_output,384*4*NBASE*4,cudaMemcpyDeviceToHost);
  FILE *fout;
  fout=fopen("test3.test","wb");
  fwrite(odata,sizeof(char),384*4*NBASE*4,fout);
  fclose(fout);*/
  
  cudaFree(d_idxs);
  free(h_idxs);
  //cudaStreamDestroy(stream);  
}




// function to copy and reorder d_input to d_r and d_i
// input is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
// output is [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
// starts by running transpose on [NPACKETS_PER_BLOCK * NANTS, NCHAN_PER_PACKET * 2 * 2] matrix in doubleComplex form.
// then fluffs using simple kernel
void reorderCorrInputCuda(dmem_corr *d) {
  
  // transpose input data
#if defined (OLD_BLAS)  
  dim3 dimBlock(32, 32), dimGrid((NCHAN_PER_PACKET*2*2)/32, ((NPACKETS_PER_BLOCK)*NANTS)/32);

  // TUNABLE
  int blockDim = 128;
  int blocks = NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/blockDim;
  transpose_matrix_char<<<dimGrid, dimBlock>>>(d->d_input, d->d_tx);
  promoteComplexCharToPlanarHalf<<<blocks, blockDim>>>(d->d_tx, (half*)d->d_r, (half*)d->d_i);
#else
  promoteComplexCharToPlanarHalf<<<blocks, blockDim>>>(d->d_input, (half*)d->d_r, (half*)d->d_i);
#endif
}


// kernels to reorder and fluff input data for beamformer
// initial data is [NPACKETS_PER_BLOCK, (NANTS/2), NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]            
// want [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, (NANTS/2), 8chan, 2 times, 2 pol, 4-bit complex]
// run as 16x16 tiled transpose with 32-byte words 
// launch with dim3 dimBlock(16, 8) and dim3 dimGrid(Width/16, Height/16)
// here, width=NCHAN_PER_PACKET/8 is the dimension of the fastest input index
// dim3 dimBlock1(16, 8), dimGrid1(NCHAN_PER_PACKET/8/16, (NPACKETS_PER_BLOCK)*(NANTS/2)/16);
void transposeInputBeamformerCuda(double *idata, double *odata, std::vector<int> &dim_block_in,
				  std::vector<int> &dim_grid_in) {

  // Create CUDA objects for launch
  dim3 dim_block(dim_block_in[0], dim_block_in[1]);
  dim3 dim_grid(dim_grid_in[0], dim_grid_in[1]);

  // Launch kernel
  transpose_input_beamformer<<<dim_grid, dim_block>>>(idata, odata);
}


// GPU-powered function to populate weights matrix for beamformer
// file format:
// sequential pairs of eastings and northings
// then [NANTS, 48, R/I] calibs

void calcWeightsCuda(dmem_bf *d) {

  // allocate
  float *antpos_e = (float *)malloc(sizeof(float)*NANTS);
  float *antpos_n = (float *)malloc(sizeof(float)*NANTS);
  float *calibs = (float *)malloc(sizeof(float)*NANTS*(NCHAN_PER_PACKET/8)*2*2);
  float *d_antpos_e, *d_antpos_n, *d_calibs;
  float wnorm;
  cudaMalloc((void **)(&d_antpos_e), sizeof(float)*NANTS);
  cudaMalloc((void **)(&d_antpos_n), sizeof(float)*NANTS);
  cudaMalloc((void **)(&d_calibs), sizeof(float)*NANTS*(NCHAN_PER_PACKET/8)*2*2);

  // deal with antpos and calibs
  //int iant;
  //int found;
  for (int i=0;i<NANTS;i++) {
    antpos_e[i] = d->h_winp[2*i];
    antpos_n[i] = d->h_winp[2*i+1];
  }
  for (int i=0;i<NANTS*(NCHAN_PER_PACKET/8)*2;i++) {

    // DEBUG CODE?
    //iant = (int)(i/((NCHAN_PER_PACKET/8)*2));
    //found = 0;
    //for (int j=0;j<d->nflags;j++)
    //if (d->flagants[j]==iant) found = 1;

    calibs[2*i] = d->h_winp[2*NANTS+2*i];
    calibs[2*i+1] = d->h_winp[2*NANTS+2*i+1];

    wnorm = sqrt(calibs[2*i]*calibs[2*i] + calibs[2*i+1]*calibs[2*i+1]);
    if (wnorm!=0.0) {
      calibs[2*i] /= wnorm;
      calibs[2*i+1] /= wnorm;
    }

    //if (found==1) {
    //calibs[2*i] = 0.;
    //calibs[2*i+1] = 0.;
    //}
  }

  //for (int i=0;i<NANTS*(NCHAN_PER_PACKET/8)*2;i++) printf("%f %f\n",calibs[2*i],calibs[2*i+1]);
  
  cudaMemcpy(d_antpos_e,antpos_e,NANTS*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_antpos_n,antpos_n,NANTS*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_calibs,calibs,NANTS*(NCHAN_PER_PACKET/8)*2*2*sizeof(float),cudaMemcpyHostToDevice);

  // run kernel to populate weights matrix
  populate_weights_matrix<<<2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)/128, 128>>>(d_antpos_e, d_antpos_n, d_calibs, (half*)d->weights_r, (half*)d->weights_i, d->d_freqs);  
  
  // free stuff
  cudaFree(d_antpos_e);
  cudaFree(d_antpos_n);
  cudaFree(d_calibs);
  free(antpos_e);
  free(antpos_n);
  free(calibs);
  
}

// kernel to fluff input bf data
// run with NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128 blocks of 128 threads
void fluffInputBeamformerCuda(char *input, void *b_real, void *b_imag, int blocks, int tpb) {

  // Launch kernel
  fluff_input_beamformer<<<blocks, tpb>>>(input, (half*)b_real, (half*)b_imag);  
}

// transpose, add and scale kernel for bf
// assume breakdown into tiles of 16x16, and run with 16x8 threads per block
// launch with dim3 dimBlock(16, 8) and dim3 dimGrid((NBEAMS/2)*(NPACKETS_PER_BLOCK/4)/16, (NCHAN_PER_PACKET/8)/16)
// scf is a per-beam scale factor to enable recasting as unsigned char
void transposeScaleBeamformerCuda(void *ir, void *ii, unsigned char *odata, std::vector<int> &dim_block_in,
				  std::vector<int> &dim_grid_in) {
  
  // Create CUDA objects for launch
  dim3 dim_block(dim_block_in[0], dim_block_in[1]);
  dim3 dim_grid(dim_grid_in[0], dim_grid_in[1]);
  
  // Launch kernel
  transpose_scale_beamformer<<<dim_grid, dim_block>>>((half*)ir, (half*)ii, odata);
}

// sum over all times in output beam array
// run with (NCHAN_PER_PACKET/8)*(NBEAMS/2) blocks of (NPACKETS_PER_BLOCK/4) threads
void sumBeamCuda(unsigned char *input, float *output, int blocks, int tpb) {

  // Launch kernel
  sum_beam<<<blocks,tpb>>>(input, output);  
}

void dsaXmemsetCuda(void *array, int ch, size_t n){
  cudaMemset(array, ch, n);
}

void dsaXmemcpyCuda(void *array_out, void *array_in, size_t n, dsaXMemcpyKind kind){
  cudaError error = cudaSuccess;
  switch(kind) {
  case dsaXMemcpyHostToHost:
    error = cudaMemcpy(array_out, array_in, n, cudaMemcpyHostToHost);
    break;
  case dsaXMemcpyHostToDevice:
   error = cudaMemcpy(array_out, array_in, n, cudaMemcpyHostToDevice);
   break;
  case dsaXMemcpyDeviceToHost:
    error = cudaMemcpy(array_out, array_in, n, cudaMemcpyDeviceToHost);
    break;
  case dsaXMemcpyDeviceToDevice:
    error = cudaMemcpy(array_out, array_in, n, cudaMemcpyDeviceToDevice);
    break;
  default:
    std::cout << "dsaX error: unknown dsaXMemcpyKind" << std::endl;
  }
  if(error != cudaSuccess) cudaGetLastError();
}

void dsaXDeviceSynchronizeCuda() {
  cudaDeviceSynchronize();
}

