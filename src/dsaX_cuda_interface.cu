#include "dsaX_cuda_interface.h"

// allocate device memory
void initialize_device_memory(dmem *d, int bf) {
  
  // for correlator
  if (bf==0) {
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

  // for beamformer
  if (bf==1) {
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
}
// deallocate device memory
void deallocate_device_memory(dmem *d, int bf) {
  
  cudaFree(d->d_input);

  if (bf==0) {
    cudaFree(d->d_r);
    cudaFree(d->d_i);
    cudaFree(d->d_tx);
    cudaFree(d->d_output);
    cudaFree(d->d_outr);
    cudaFree(d->d_outi);
    cudaFree(d->d_tx_outr);
    cudaFree(d->d_tx_outi);
  }
  if (bf==1) {
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
}

// function to copy d_outr and d_outi to d_output
// inputs are [NCHAN_PER_PACKET, 2 time, 2 pol, NANTS, NANTS]
// the corr matrices are column major order
// output needs to be [NBASE, NCHAN_PER_PACKET, 2 pol, 2 complex]
// start with transpose to get [NANTS*NANTS, NCHAN_PER_PACKET*2*2], then sum into output using kernel
void reorder_output_device(dmem * d) {
  
  // transpose input data
  dim3 dimBlock(32, 8), dimGrid((NANTS*NANTS)/32,(NCHAN_PER_PACKET*2*2*halfFac)/32);
  transpose_matrix<<<dimGrid,dimBlock>>>(d->d_outr,d->d_tx_outr);
  transpose_matrix<<<dimGrid,dimBlock>>>(d->d_outi,d->d_tx_outi);

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
  cudaMemcpy(d_idxs,h_idxs,sizeof(int)*NBASE,cudaMemcpyHostToDevice);

  // run kernel to finish things
  corr_output_copy<<<NCHAN_PER_PACKET*2*NBASE/128,128>>>(d->d_tx_outr,d->d_tx_outi,d->d_output,d_idxs);

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

// kernel to fluff input
// run with 128 threads and NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/128 blocks
__global__ void corr_input_copy(char *input, half *inr, half *ini) {

  int bidx = blockIdx.x;  // assume NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/128
  int tidx = threadIdx.x; // assume 128 threads per block
  int iidx = bidx*128+tidx;

  // 15 in unsigned char binary is 00001111. Perform bitwise & on 15 and input char data iiiirrrr
  // to get real part 4 bit data.
  // 0000rrrr
  // Bit shift this result by 4 to the left.
  // rrrr0000
  // Cast to signed char.
  // +-rrr0000
  // Bitshift mantisa only to the right by 4 bits
  // +-0000rrr
  // Cast to float and use CUDA intrinsic to cast to signed half
  inr[iidx] = __float2half((float)((char)((   (unsigned char)(input[iidx]) & (unsigned char)(15)  ) << 4) >> 4));

  // 240 in unsigned char binary is 11110000. Perform bitwise & on 240 and input char data iiiirrrr
  // to get imag part 4 bit data
  // iiii0000.
  // Cast to signed char
  // +-iii0000
  // Bitshift mantisa only to the right by 4 bits
  // +-0000iii
  // Cast to float and use CUDA intrinsic to cast to signed half
  ini[iidx] = __float2half((float)((char)((   (unsigned char)(input[iidx]) & (unsigned char)(240)  )) >> 4));

  // Both results should be half (FP16) integers between -8 and 7.
  half re = inr[iidx];
  half im = ini[iidx];
  half lim = 2.;
  if( (re > lim || re < -lim) || (im > lim || im < -lim)) {
    //printf("re = %f, im = %f\n", __half2float(re), __half2float(im));
  }
}

// transpose kernel
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width is the dimension of the fastest index
template <typename in_prec, typename out_prec> __global__ void transpose_matrix(in_prec * idata, out_prec * odata) {

  __shared__ in_prec tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

}


// function to copy and reorder d_input to d_r and d_i
// input is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
// output is [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
// starts by running transpose on [NPACKETS_PER_BLOCK * NANTS, NCHAN_PER_PACKET * 2 * 2] matrix in doubleComplex form.
// then fluffs using simple kernel
void reorder_input_device(char *input, char * tx, half *inr, half *ini) {

  // transpose input data
  dim3 dimBlock(32, 8), dimGrid((NCHAN_PER_PACKET*2*2)/32, ((NPACKETS_PER_BLOCK)*NANTS)/32);
  transpose_matrix<<<dimGrid,dimBlock>>>(input, tx);
  corr_input_copy<<<NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/128, 128>>>(tx, inr, ini);
}

// kernel to help with reordering output
// outr and outi are [NANTS, NANTS, NCHAN_PER_PACKET, 2time, 2pol, halfFac]
// run with NCHAN_PER_PACKET*2*NBASE/128 blocks of 128 threads
__global__ void corr_output_copy(half *outr, half *outi, float *output, int *indices_lookup) {

  int bidx = blockIdx.x; // assume NCHAN_PER_PACKET*2*NBASE/128
  int tidx = threadIdx.x; // assume 128
  int idx = bidx*128+tidx;
  
  int baseline = (int)(idx / (NCHAN_PER_PACKET * 2));
  int chpol = (int)(idx % (NCHAN_PER_PACKET * 2));
  int ch = (int)(chpol / 2);
  int base_idx = indices_lookup[baseline];
  int iidx = base_idx * NCHAN_PER_PACKET + ch;
  int pol = (int)(chpol % 2);

  float v1=0., v2=0.;

  // Use CUDA casting intrinsic __half2float
  for (int i=0;i<halfFac;i++) {
    v1 += __half2float(outr[(4*iidx+pol)*halfFac+i])+__half2float(outr[(4*iidx+2+pol)*halfFac+i]);
    v2 += __half2float(outi[(4*iidx+pol)*halfFac+i])+__half2float(outi[(4*iidx+2+pol)*halfFac+i]);
  }

  output[2*idx] = v1;
  output[2*idx+1] = v2;
  
}

// kernels to reorder and fluff input data for beamformer
// initial data is [NPACKETS_PER_BLOCK, (NANTS/2), NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]            
// want [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, (NANTS/2), 8chan, 2 times, 2 pol, 4-bit complex]
// run as 16x16 tiled transpose with 32-byte words 
// launch with dim3 dimBlock(16, 8) and dim3 dimGrid(Width/16, Height/16)
// here, width=NCHAN_PER_PACKET/8 is the dimension of the fastest input index
// dim3 dimBlock1(16, 8), dimGrid1(NCHAN_PER_PACKET/8/16, (NPACKETS_PER_BLOCK)*(NANTS/2)/16);
__global__ void transpose_input_bf(double *idata, double *odata) {

  __shared__ double tile[16][17][4];
  
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  int width = gridDim.x * 16;

  for (int j = 0; j < 16; j += 8) {
    tile[threadIdx.y+j][threadIdx.x][0] = idata[4*((y+j)*width + x)];
    tile[threadIdx.y+j][threadIdx.x][1] = idata[4*((y+j)*width + x)+1];
    tile[threadIdx.y+j][threadIdx.x][2] = idata[4*((y+j)*width + x)+2];
    tile[threadIdx.y+j][threadIdx.x][3] = idata[4*((y+j)*width + x)+3];
  }
  
  __syncthreads();

  x = blockIdx.y * 16 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 16 + threadIdx.y;
  width = gridDim.y * 16;

  for (int j = 0; j < 16; j += 8) {
    odata[4*((y+j)*width + x)] = tile[threadIdx.x][threadIdx.y + j][0];
    odata[4*((y+j)*width + x)+1] = tile[threadIdx.x][threadIdx.y + j][1];
    odata[4*((y+j)*width + x)+2] = tile[threadIdx.x][threadIdx.y + j][2];
    odata[4*((y+j)*width + x)+3] = tile[threadIdx.x][threadIdx.y + j][3];
  }

}

// kernel to populate an instance of weights matrix [2, (NCHAN_PER_PACKET/8), NBEAMS/2, 4times*(NANTS/2)*8chan*2tim*2pol]
// run with 2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)/128 blocks of 128 threads
__global__ void populate_weights_matrix(float * antpos_e, float * antpos_n, float * calibs, half * wr, half * wi, float * fqs) {

  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int inidx = bidx*128+tidx;  
  
  // 2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)
  
  // get indices
  int iArm = (int)(inidx / ((NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)));
  int iidx = (int)(inidx % ((NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)));
  int fq = (int)(iidx / (128*(NANTS/2)*(NBEAMS/2)));
  int idx = (int)(iidx % (128*(NANTS/2)*(NBEAMS/2)));
  int bm = (int)(idx / (128*(NANTS/2)));
  int tactp = (int)(idx % (128*(NANTS/2)));
  //int t = (int)(tactp / (32*(NANTS/2)));
  int actp = (int)(tactp % (32*(NANTS/2)));
  int a = (int)(actp / 32);
  int ctp = (int)(actp % 32);
  //int c = (int)(ctp / 4);
  int tp = (int)(ctp % 4);
  //int t2 = (int)(tp / 2);
  int pol = (int)(tp % 2);
  int widx = (a+48*iArm)*(NCHAN_PER_PACKET/8)*2*2 + fq*2*2 + pol*2;
  
  // calculate weights
  float theta, afac, twr, twi;
  if (iArm==0) {
    theta = sep*(127.-bm*1.)*PI/10800.; // radians
    afac = -2.*PI*fqs[fq]*theta/CVAC; // factor for rotate
    twr = cos(afac*antpos_e[a+48*iArm]);
    twi = sin(afac*antpos_e[a+48*iArm]);
    wr[inidx] = __float2half((twr*calibs[widx] - twi*calibs[widx+1]));
    wi[inidx] = __float2half((twi*calibs[widx] + twr*calibs[widx+1]));
    //wr[inidx] = __float2half(calibs[widx]);
    //wi[inidx] = __float2half(calibs[widx+1]);
  }
  if (iArm==1) {
    theta = sep*(127.-bm*1.)*PI/10800.; // radians
    afac = -2.*PI*fqs[fq]*theta/CVAC; // factor for rotate
    twr = cos(afac*antpos_n[a+48*iArm]);
    twi = sin(afac*antpos_n[a+48*iArm]);
    wr[inidx] = __float2half((twr*calibs[widx] - twi*calibs[widx+1]));
    wi[inidx] = __float2half((twi*calibs[widx] + twr*calibs[widx+1]));
    //wr[inidx] = __float2half(calibs[widx]);
    //wi[inidx] = __float2half(calibs[widx+1]);
  }
    
}

// GPU-powered function to populate weights matrix for beamformer
// file format:
// sequential pairs of eastings and northings
// then [NANTS, 48, R/I] calibs

void calc_weights(dmem *d) {

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
  populate_weights_matrix<<<2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)/128,128>>>(d_antpos_e,d_antpos_n,d_calibs,d->weights_r,d->weights_i,d->d_freqs);  
  
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
__global__ void fluff_input_bf(char * input, half * dr, half * di) {

  int bidx = blockIdx.x; // assume NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128
  int tidx = threadIdx.x; // assume 128
  int idx = bidx*128+tidx;

  dr[idx] = __float2half(0.015625*((float)((char)(((unsigned char)(input[idx]) & (unsigned char)(15)) << 4) >> 4)));
  di[idx] = __float2half(0.015625*((float)((char)(((unsigned char)(input[idx]) & (unsigned char)(240))) >> 4)));

  // Both results should be half (FP16) integers between -8 and 7.
  //half re = dr[idx];
  //half im = di[idx];
  //half lim = 0;
  //if( (re > lim || re < -lim) || (im > lim || im < -lim)) {
  //printf("re = %f, im = %f\n", __half2float(re), __half2float(im));
  //}

  
}

// transpose, add and scale kernel for bf
// assume breakdown into tiles of 16x16, and run with 16x8 threads per block
// launch with dim3 dimBlock(16, 8) and dim3 dimGrid((NBEAMS/2)*(NPACKETS_PER_BLOCK/4)/16, (NCHAN_PER_PACKET/8)/16)
// scf is a per-beam scale factor to enable recasting as unsigned char
__global__ void transpose_scale_bf(half * ir, half * ii, unsigned char * odata) {

  __shared__ float tile[16][17];
  
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  int width = gridDim.x * 16;
  float dr, di;

  for (int j = 0; j < 16; j += 8) {
    dr = (float)(ir[(y+j)*width + x]);
    di = (float)(ii[(y+j)*width + x]);
    tile[threadIdx.y+j][threadIdx.x] = (dr*dr+di*di);
  }

  __syncthreads();

  x = blockIdx.y * 16 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 16 + threadIdx.y;
  width = gridDim.y * 16;

  for (int j = 0; j < 16; j += 8)
    odata[(y+j)*width + x] = (unsigned char)(tile[threadIdx.x][threadIdx.y + j]/128.);

}

// sum over all times in output beam array
// run with (NCHAN_PER_PACKET/8)*(NBEAMS/2) blocks of (NPACKETS_PER_BLOCK/4) threads
__global__ void sum_beam(unsigned char * input, float * output) {

  __shared__ float summ[512];
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  //int idx = bidx*256+tidx;
  int bm = (int)(bidx/48);
  int ch = (int)(bidx % 48);

  summ[tidx] = (float)(input[bm*256*48 + tidx*48 + ch]);

  __syncthreads();

  if (tidx<256) {
    summ[tidx] += summ[tidx+256];
    summ[tidx] += summ[tidx+128];
    summ[tidx] += summ[tidx+64];
    summ[tidx] += summ[tidx+32];
    summ[tidx] += summ[tidx+16];
    summ[tidx] += summ[tidx+8];
    summ[tidx] += summ[tidx+4];
    summ[tidx] += summ[tidx+2];
    summ[tidx] += summ[tidx+1];
  }

  if (tidx==0) output[bidx] = summ[tidx];
  
}
