#pragma once

#include "dsaX_cuda_headers.h"

__device__ void inspectPackedDataInKernel(char input, int i) {
  float re = (float)((char)((   (unsigned char)(input) & (unsigned char)(15)  ) << 4) >> 4);
  float im = (float)((char)((   (unsigned char)(input) & (unsigned char)(240))) >> 4);
  
  if(re != 0 || im != 0) printf("val[%d] = (%f,%f)\n", i, re, im);
}

// KERNELS
// DMH: Abstract hardcoded launch parameters
__global__ void transpose_input_beamformer(double *idata, double *odata) {
  
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

// kernel to help with reordering output
// outr and outi are [NANTS, NANTS, NCHAN_PER_PACKET, 2time, 2pol, halfFac]
// run with NCHAN_PER_PACKET*2*NBASE/128 blocks of 128 threads
__global__ void corr_output_copy(half *outr, half *outi, float *output, int *indices_lookup) {
  
  int bidx = blockIdx.x; // assume NCHAN_PER_PACKET*2*NBASE/128
  int tidx = threadIdx.x; // assume 128
  int idx = blockDim.x * bidx + tidx;
  
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

// transpose kernel
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width is the dimension of the fastest index
template <typename in_prec, typename out_prec> __global__ void transpose_matrix(in_prec * idata, out_prec * odata) {
  
  __shared__ in_prec tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8) {
    tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
    //inspectPackedDataInKernel(idata[(y+j)*width + x], (y+j)*width + x);
  }
  
  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];

}

// transpose kernel
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
// here, width is the dimension of the fastest index
__global__ void transpose_matrix_char(char * idata, char * odata) {
  
  __shared__ char tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int width = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8) {
    tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
    //inspectPackedDataInKernel(idata[(y+j)*width + x], (y+j)*width + x);
  }
  
  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  width = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8) {
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}


// kernel to fluff input
// run with 128 threads and NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4/128 blocks
__global__ void corr_input_copy(char *input, half *inr, half *ini) {

  int bidx = blockIdx.x;  
  int tidx = threadIdx.x; 
  int iidx = blockDim.x * bidx + tidx;
  
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

  //if(__half2float(inr[iidx]) != 0 || __half2float(ini[iidx]) != 0) printf("corr_input_copy %i = (%f,%f)\n", iidx, __half2float(inr[iidx]), __half2float(ini[iidx]));
}

// kernel to populate an instance of weights matrix
// [2, (NCHAN_PER_PACKET/8), NBEAMS/2, 4times*(NANTS/2)*8chan*2tim*2pol]
// run with 2*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*128*(NANTS/2)/128 blocks of 128 threads
// TUNABLE
__global__ void populate_weights_matrix(float * antpos_e, float * antpos_n, float * calibs, half * wr, half * wi, float * fqs) {
  
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int inidx = 128 * bidx + tidx;  
  
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

// kernel to fluff input bf data
// run with NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128 blocks of 128 threads
__global__ void fluff_input_beamformer(char * input, half * dr, half * di) {
  
  int bidx = blockIdx.x; 
  int tidx = threadIdx.x;
  int idx = blockDim.x * bidx + tidx;
  
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
__global__ void transpose_scale_beamformer(half * ir, half * ii, unsigned char * odata) {

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
__global__ void sum_beam(unsigned char *input, float *output) {
  
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
