// -*- c++ -*-       
/* will implement the 64-input beamformer 

does N beams of 256

order is (taking time as 8x 8.192e-6) 
[2048 time, 63 antennas, 768 channels, 2 pol, r/i]
Load in 16 times at a time, so that we have (in units of what needs to be added)
[16 time, 63 antennas, 96 channels, 8 chunnels, 2 pol, r/i]

This should be reordered on the cpu to 
[16 time, 96 channels, 63 antennas, 8 chunnels, 2 pol, r/i]

The first kernel, launched with 1536 blocks of 64 threads, needs to
 - promote each measurement and store in shared mem, parallelizing over ants. need only 8 kB. 
 - each thread processes 4 beams, adding everything. for each beam,
  + for each chunnel and pol, calculate weights using cal weights and ant positions, 
  + add everything into output array
Output array has order [beam, 96 frequency, 16 time]

Shared mem requirement: 8 kB for promoted data, 512b for positions, nch*1024b for weights

Initialy we start with 4-bit numbers. these are first rotated using 17-bit weights, yielding 22-bit numbers. 
these are then added: (64 ant)^2 * (2 complex) * (32 chan) * (2 pol) * (16 time). 
after adding by 64 ants, we have 28-bit numbers. Need to bit shift right by 19 after adding 64 ants. This will yield 29-bit numbers. Need to bit shift right by 21 to pick off lowest 8 bits. 

Do everything in floating point until second kernel. 

Second kernel will simply add times and adjacent channels and pick leading 8 bits
Then copy back to specific locations in host to form final [beam, time, frequency] array, to be sent to corner turn.

 */
#include <iostream>
#include <algorithm>
using std::cout;
using std::cerr;
using std::endl;
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>
#include <syslog.h>
#include <pthread.h>

#include <mma.h>
#include <cuda.h>
#include "cuda_fp16.h"
//#include "dada_cuda.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_def.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cuda_runtime_api.h>
using namespace nvcuda;

// global variables
int DEBUG = 0;


// kernel for summing and requantizing
// input array has order [beam, 48 frequency, 2 pol, 16 time]
// need to output to [4 time, beam, 48 frequency]
// bp is scale factor for each beam 
// run with 256*48=12288 blocks and 32 threads
__global__
void adder(float *input, unsigned char *output, float *bp) {

  // get block and thread ids
  int bidx = blockIdx.x; // assume 256*48=12288
  int tidx = threadIdx.x; // assume 32
  //int fidx = 2*(bidx % 24);
  int beamidx = (int)(bidx / 48);
  
  // declare shared mem
  __shared__ float data[32]; // data block to be summed  

  // transfer from input to shared mem
  data[tidx] = input[bidx*32];

  // sync
  __syncthreads();

  // complete sum
  if (tidx<16) {
    data[tidx] += data[tidx+16]; // over pols

    data[tidx] += data[tidx+2];
    data[tidx] += data[tidx+1];
  }
  // now tidx = 0, 4, 8, 12 are what we want! 

  __syncthreads();
  
  // store
  if (tidx == 0) 
    output[bidx] = (unsigned char)(__float2int_rn(data[0]*bp[beamidx])/2);
  if (tidx == 4) 
    output[bidx + 12288] = (unsigned char)(__float2int_rn(data[4]*bp[beamidx])/2);
  if (tidx == 8) 
    output[bidx + 2*12288] = (unsigned char)(__float2int_rn(data[8]*bp[beamidx])/2);
  if (tidx == 12) 
    output[bidx + 3*12288] = (unsigned char)(__float2int_rn(data[12]*bp[beamidx])/2);
      
}

// kernel for promotion
/*
orig input is [16 time, NANT antennas, 48 channels, 16 chunnels, 2 pol, r/i]
input is [16 time, 48 channels, NANT antennas, 16 chunnels, 2 pol, r/i]
output needs to be [16 time, 48 channels, 2 pol, 64 antennas, 16 chunnels, r/i] 
promoted to half precision  

launch with 16*48*NANT blocks of 32 threads

 */
__global__ void promoter(char *input, half *inr, half *ini) {

  int bidx = blockIdx.x; // assume 16*48*NANT
  int tidx = threadIdx.x; // assume 32
  int iidx = bidx*32+tidx;
  int pol = (int)(tidx % 2);
  int chunnel = (int)(tidx / 2);
  
  /*int ant = (int)(bidx % NANT);
  int time_chan = (int)(bidx / NANT);    
  int oidx = time_chan*2048+pol*1024+ant*16+chunnel;*/

  int chan = (int)(bidx % 48);
  int time_ant = (int)(bidx / 48);
  int tim = (int)(time_ant / NANT);
  int ant = (int)(time_ant % NANT);
  int oidx = tim*98304 + chan*2048 + pol*1024 + ant*16 + chunnel;

  inr[oidx] = __float2half((float)(((char)((input[iidx] & 15) << 4)) >> 4));
  ini[oidx] = __float2half((float)(((char)((input[iidx] & 240))) >> 4));

}

// 16 time, 48 channels, 2 pol, 64 antennas, 16 chunnels
// for first time, launch with 3072, 32
__global__ void printer(half *inr, half *ini) {

  int idx = blockIdx.x*32+threadIdx.x;
  float ir = __half2float(inr[idx]);
  float ii = __half2float(ini[idx]);

  int chunnel = (int)(threadIdx.x % 16);
  int channel = (int)(blockIdx.x/64);
  int tt = (int)(blockIdx.x % 64);
  int pol = (int)(tt/32);
  int ant = ((int)(tt % 32))*((int)(threadIdx.x / 16));
  
  if (ir!=0. || ii!=0.) {
    printf("%d %d %d %d %f %f\n",channel,pol,ant,chunnel,ir,ii);
  }
  
}


// kernel for beamforming
/*

Assumes that up to NANT antennas (nominally 63) are populated. 

Input is [16 time, 48 channels, 2 pol, 64 antennas, 16 chunnels, r/i] (promoted)

Arithmetic... for rotation, d2r = wr*dr-wi*di; d2i = wi*dr+wr*di

Conventions for beamforming. beam 0 is furthest East, beam 127 is at meridian. antpos (D) is easting. 
for bf weight calculation, where theta = s(127-n), ang = 2*pi*nu*theta*D/c; wr = cos(ang), wi = sin(ang)
use __float2int_rn, cosf, sinf intrinsics. 

Each warp (==block) has to deal with 256 beams for 64 ants, summing over 16 chunnels and pols. 
Do it in tiles of 16 beams and 16 ants for 

Output array has order [beam, 48 frequency, 2 pol, 16 time]

inr and ini are data, in [16 time, 48 freq, 2 pol, 64 ant, 16 chunnels] for real and imag
wr and wi are weights, in [48 freq, 2 pol, 16 beam_tile, 4 ant_tile, 16 beam, 16 ant]

launch with 16time * 48freq * 2pol * 16beam_tile blocks of 32 threads for massive utilization
 = 24576 blocks

*/
__global__ void beamformer(half *inr, half *ini, half *wr, half *wi, float *output, int stuffants) {

  // get block and thread ids
  int bidx = blockIdx.x; // assume 24576
  int tidx = threadIdx.x; // assume 32
  int orig_bidx = (int)(bidx / 16);
  int beam_tile = (int)(bidx % 16);
  int stuff_tile = (int)(beam_tile % 4);
  int data_offset = orig_bidx*1024; // offset for first part of data
  int weight_offset = (int)(orig_bidx % 96); // offset for first part of weight
  weight_offset *= 16384;
  int idx1, idx2;
  int f_idx = (int)(orig_bidx % 96);
  int tim_idx = (int)(orig_bidx / 96);
  int oidx = f_idx*16 + tim_idx;
  
  // shared memory for convenience
  __shared__ float summr[16][16]; // beam, chunnel
  __shared__ float summi[16][16]; // beam, chunnel
  
  // accumulate real and imag parts into [16 beam x 16 f] fragments
  // Declare the fragments.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> wr_inr_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> wr_ini_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> wi_inr_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> wi_ini_frag;
  
  // zero out accumulators
  wmma::fill_fragment(wr_inr_frag, 0.0f);
  wmma::fill_fragment(wr_ini_frag, 0.0f);
  wmma::fill_fragment(wi_inr_frag, 0.0f);
  wmma::fill_fragment(wi_ini_frag, 0.0f);

  if (stuffants) {        

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> c_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> d_frag;
    wmma::load_matrix_sync(c_frag, inr + data_offset + stuff_tile*256, 16);
    wmma::load_matrix_sync(d_frag, inr + data_offset + stuff_tile*256, 16);
    wmma::mma_sync(wr_inr_frag, c_frag, d_frag, wr_inr_frag);
    wmma::load_matrix_sync(c_frag, ini + data_offset + stuff_tile*256, 16);
    wmma::load_matrix_sync(d_frag, ini + data_offset + stuff_tile*256, 16);
    wmma::mma_sync(wr_inr_frag, c_frag, d_frag, wr_inr_frag);
    
  }
  else {
  
    // loop over ant tiles
    for (int ant_tile=0; ant_tile<4; ant_tile++) {
      
      // copy weight and data to fragments, and multiply to accumulators
      
      wmma::load_matrix_sync(a_frag, wr + weight_offset + beam_tile*1024 + ant_tile*256, 16);
      wmma::load_matrix_sync(b_frag, inr + data_offset + ant_tile*256, 16);
      wmma::mma_sync(wr_inr_frag, a_frag, b_frag, wr_inr_frag);
      
      wmma::load_matrix_sync(a_frag, wi + weight_offset + beam_tile*1024 + ant_tile*256, 16);
      wmma::mma_sync(wi_inr_frag, a_frag, b_frag, wi_inr_frag);
      
      wmma::load_matrix_sync(b_frag, ini + data_offset + ant_tile*256, 16);
      wmma::mma_sync(wi_ini_frag, a_frag, b_frag, wi_ini_frag);
      
      wmma::load_matrix_sync(a_frag, wr + weight_offset + beam_tile*1024 + ant_tile*256, 16);
      wmma::mma_sync(wr_ini_frag, a_frag, b_frag, wr_ini_frag);
      
    }

    // form real and imaginary matrices
    for(int i=0; i < wr_inr_frag.num_elements; i++) {
      wr_inr_frag.x[i] = wr_inr_frag.x[i] - wi_ini_frag.x[i]; // output real
      wi_inr_frag.x[i] = wi_inr_frag.x[i] + wr_ini_frag.x[i]; // output imag
      wr_inr_frag.x[i] = wr_inr_frag.x[i]*wr_inr_frag.x[i] + wi_inr_frag.x[i]*wi_inr_frag.x[i]; // squared
    }
  }

  // at this stage the matrices are [beam, chunnel], and need to be summed over columns
    
  // copy back to shared mem
  float *p1;
  p1 = &summr[0][0];
  wmma::store_matrix_sync(p1, wr_inr_frag, 16, wmma::mem_row_major);

  if (!stuffants) {
  
    // do thread reduction for each beam
    if (tidx<8) {
      for (int i=0;i<4;i++) summr[i][tidx] += summr[i][tidx+8];
      for (int i=0;i<4;i++) summr[i][tidx] += summr[i][tidx+4];
      for (int i=0;i<4;i++) summr[i][tidx] += summr[i][tidx+2];
      for (int i=0;i<4;i++) summr[i][tidx] += summr[i][tidx+1];
    }
    if (tidx>=8 && tidx<16) {
      for (int i=4;i<8;i++) summr[i][tidx-8] += summr[i][tidx+8-8];
      for (int i=4;i<8;i++) summr[i][tidx-8] += summr[i][tidx+4-8];
      for (int i=4;i<8;i++) summr[i][tidx-8] += summr[i][tidx+2-8];
      for (int i=4;i<8;i++) summr[i][tidx-8] += summr[i][tidx+1-8];  
    }
    if (tidx>=16 && tidx<24) {
      for (int i=8;i<12;i++) summr[i][tidx-16] += summr[i][tidx+8-16];
      for (int i=8;i<12;i++) summr[i][tidx-16] += summr[i][tidx+4-16];
      for (int i=8;i<12;i++) summr[i][tidx-16] += summr[i][tidx+2-16];
      for (int i=8;i<12;i++) summr[i][tidx-16] += summr[i][tidx+1-16];  
    }
    if (tidx>=24) {
      for (int i=12;i<16;i++) summr[i][tidx-24] += summr[i][tidx+8-24];
      for (int i=12;i<16;i++) summr[i][tidx-24] += summr[i][tidx+4-24];
      for (int i=12;i<16;i++) summr[i][tidx-24] += summr[i][tidx+2-24];
      for (int i=12;i<16;i++) summr[i][tidx-24] += summr[i][tidx+1-24];  
    }

    __syncthreads();
    
    // now summr[beam][0] can go into output
    if (tidx<16) {
      output[(beam_tile*16+tidx)*1536 + oidx] = summr[tidx][0];
    }

  }
  else {

    if (tidx<16) {
      output[(beam_tile*16+tidx)*1536 + oidx] = summr[tidx][tidx];
    }


  }

  
}

// kernel to calculate weights - needed because weights are halfs
// launch with 256 threads in 6144 blocks
__global__
void calc_weights(float *antpos, float *weights, float *freqs, half *wr, half *wi) {

  // assume 256 threads in 6144 blocks
  int bidx = blockIdx.x; // over 48f, 2pol, 16 beam_tile, 4 ant_tile
  int tidx = threadIdx.x;
  int f = (int)(bidx / 128);
  int cc = (int)(bidx % 128);
  int pol = (int)(cc / 64);
  cc = (int)(cc % 64);
  int beam_tile = (int)(cc / 4);
  int ant_tile = (int)(cc % 4);
  int beam_i = (int)(tidx / 16);
  int ant_i = (int)(tidx % 16);

  int beam = beam_tile*16+beam_i;
  int ant = ant_tile*16+ant_i;
  int i = bidx*256+tidx;
  int widx = ant*NW*2*2 + f*2*2 + pol*2;
  
  float theta = sep*(127.-beam*1.)*PI/10800.; // radians
  float afac = -2.*PI*freqs[f*8+4]*theta/CVAC; // factor for rotate
  float twr = cos(afac*antpos[ant]);
  float twi = sin(afac*antpos[ant]);

  wr[i] = __float2half((twr*weights[widx] - twi*weights[widx+1]));
  wi[i] = __float2half((twi*weights[widx] + twr*weights[widx+1]));
  
  
}  
 
  
// function prototypes
void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);
int dada_bind_thread_to_core (int core);
int init_weights(char *fnam, float *antpos, float *weights, char *flagants);
void reorder_block(char *block);
void calc_bp(float *data, float *bp, int pr);


// performs massive summation to calculate bp
// input array has order [beam, 96 frequency, 16 time]
// bp has size 48 - no way to avoid strided memory access
// returns factor to correct data
void calc_bp(float *data, float *bp, int pr) {

  int i=0;
  
  for (int b=0;b<256;b++) {
    for (int f=0;f<48;f++) {
      for (int a=0;a<32;a++) {
	bp[b] += data[i];
	if (pr && data[i]!=0.) printf("%d %d %d %f\n",b,f,a,data[i]);
	i++;
      }
    }
  }

}

// performs cpu reorder of block to be loaded to GPU
void reorder_block(char * block) {

  // from [16 time, NANT antennas, 48 channels, 16 chunnels, 2 pol, r/i]
  // to [16 time, 48 channels, NANT antennas, 16 chunnels, 2 pol, r/i]
  // 24576*NANT in total. 1536*NANT per time
  
  char * output = (char *)malloc(sizeof(char)*24576*NANT);
  
  for (int i=0;i<16;i++) { // over time
    for (int j=0;j<NANT;j++) { // over ants
      for (int k=0;k<48;k++) { // over channels

	// copy 32 bytes
	memcpy(output + i*1536*NANT + k*NANT*32 + j*32, block + i*1536*NANT + j*1536 + k*32, 32); 
	
      }
    }
  }

  memcpy(block,output,24576*NANT);
  free(output);

}


// loads in weights
int init_weights(char * fnam, float *antpos, float *weights, char *flagants) {

  // assumes 64 antennas
  // antpos: takes only easting
  // weights: takes [ant, NW==48] 

  FILE *fin;
  FILE *fants;
  
  if (!(fin=fopen(fnam,"rb"))) {
    syslog(LOG_ERR,"Couldn't open weights file %s",fnam);
    return 1;
  }
  if (!(fants=fopen(flagants,"r"))) {
    syslog(LOG_ERR,"Couldn't open flag ants file %s",flagants);
    return 1;
  }

  fread(antpos,64*sizeof(float),1,fin);
  fread(weights,64*NW*2*2*sizeof(float),1,fin);
  float wnorm;
  for (int i=0;i<64*NW*2;i++) {
    wnorm = sqrt(weights[2*i]*weights[2*i] + weights[2*i+1]*weights[2*i+1]);
    if (wnorm!=0.0) {
      weights[2*i] /= wnorm;
      weights[2*i+1] /= wnorm;
    }
  }
	

  int ant;
  while (!feof(fants)) {
    fscanf(fants,"%d\n",&ant);
    for (int j=0;j<NW*2*2;j++) {
      weights[ant*NW*2*2+j] = 0.0;
    }
  }
      
  fclose(fants);
  fclose(fin);
  if (DEBUG) syslog(LOG_INFO,"Loaded antenna positions and weights");
  return 0;

}


void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out)
{

  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);

  if (dada_hdu_unlock_write (out) < 0)
    {
      syslog(LOG_ERR, "could not unlock write on hdu_out");
    }
  dada_hdu_destroy (out);
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_beamformer [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -f filename for antenna stuff [no default]\n"
	   " -i input key [default REORDER_BLOCK_KEY2]\n"
	   " -o output key [default BF_BLOCK_KEY]\n"
	   " -z fch1 in MHz [default 1530]\n"
	   " -a flagants file\n"
	   " -s stuffants \n"
	   " -h print usage\n");
}

// MAIN

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_beamformer", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  // device properties
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    syslog(LOG_INFO,"Device Number: %d", i);
    syslog(LOG_INFO,"  Device name: %s", prop.name);
    syslog(LOG_INFO,"  Memory Clock Rate (KHz): %d",prop.memoryClockRate);
  }
  cudaSetDevice(1);
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t in_key = REORDER_BLOCK_KEY2;
  key_t out_key = BF_BLOCK_KEY;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  int stuffants=0;
  float fch1 = 1530.0;
  char * fnam;
  fnam=(char *)malloc(sizeof(char)*100);
  sprintf(fnam,"nofile");  
  char * flagants;
  flagants=(char *)malloc(sizeof(char)*100);
  sprintf(flagants,"nofile");  

  while ((arg=getopt(argc,argv,"c:f:i:o:z:a:sdh")) != -1)
    {
      switch (arg)
	{
	case 'c':
	  if (optarg)
	    {
	      core = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-c flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'i':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &in_key) != 1) {
		syslog(LOG_ERR, "could not parse key from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-i flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'o':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &out_key) != 1) {
		syslog(LOG_ERR, "could not parse key from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-o flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'f':
	  if (optarg)
	    {
	      strcpy(fnam,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }	  
	case 'a':
	  if (optarg)
	    {
	      strcpy(flagants,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-a flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }	  
	case 'z':
	  if (optarg)
	    {
	      fch1 = atof(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-z flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }	  
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 's':
	  stuffants=1;
	  syslog (LOG_INFO, "Will place antennas in output");
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // print stuff
  syslog(LOG_INFO,"Forming 256 beams with sep %g arcmin, fch1 %g",sep,fch1);
  syslog(LOG_INFO,"Using calibrations file %s",fnam);
  syslog(LOG_INFO,"Using flagants file %s",flagants);

  // load in weights and antpos
  float * antpos = (float *)malloc(sizeof(float)*64); // easting
  float * weights = (float *)malloc(sizeof(float)*64*NW*2*2); // complex weights [ant, NW, pol, r/i]
  float * freqs = (float *)malloc(sizeof(float)*384); // freq
  for (int i=0;i<384;i++) freqs[i] = (fch1 - i*250./8192.)*1e6;  
  
  // Bind to cpu core
  if (core >= 0)
    {
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
      syslog(LOG_NOTICE,"bound to core %d", core);
    }
  
  // DADA stuff
  
  syslog (LOG_INFO, "creating in and out hdus");
  
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to dada buffer in");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to dada buffer in");
    return EXIT_FAILURE;
  }

  hdu_out  = dada_hdu_create ();
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog (LOG_ERR,"could not connect to output  buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write(hdu_out) < 0) {
    syslog (LOG_ERR, "could not lock to output buffer");
    return EXIT_FAILURE;
  }
  
  uint64_t header_size = 0;

  // deal with headers
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "could not read next header");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %llu %llu\n",block_size,block_out);
  uint64_t  bytes_read = 0;
  int nints = NPACKETS / 16;
  uint64_t nbytes_per_int = block_size / nints;
  uint64_t nbytes_per_out = block_out / nints;
  char * block;
  unsigned char * output_buffer;
  output_buffer = (unsigned char *)malloc(sizeof(unsigned char)*block_out);
  memset(output_buffer,0,block_out);
  uint64_t written, block_id;
  
  // allocate host and device memory for calculations
  //inr and ini are data, in [16 time, 48 freq, 2 pol, 64 ant, 16 chunnels] for real and imag
  //wr and wi are weights, in [48 freq, 2 pol, 16 beam_tile, 4 ant_tile, 16 beam, 16 ant]        
  char *d_indata[NSTREAMS];
  unsigned char *d_outdata[NSTREAMS];
  float *d_transfer[NSTREAMS], *d_bp, *d_antpos, *d_weights, *d_freqs;
  half *d_wr, *d_wi, *d_inr[NSTREAMS], *d_ini[NSTREAMS];
  cudaMalloc((void **)&d_antpos, 64*sizeof(float)); // ant positions
  cudaMalloc((void **)&d_weights, 64*NW*2*2*sizeof(float)); // weights
  cudaMalloc((void **)&d_freqs, 384*sizeof(float)); // freqs        
  cudaMalloc((void **)&d_bp, 256*sizeof(float)); // bandpass
  cudaMalloc((void **)&d_wr, 48*2*16*4*16*16*sizeof(half)); // real weight
  cudaMalloc((void **)&d_wi, 48*2*16*4*16*16*sizeof(half)); // imag weight
  cudaMemcpy(d_freqs, freqs, 384*sizeof(float), cudaMemcpyHostToDevice);
  
  float *h_transfer = (float *)malloc(sizeof(float)*256*96*16*NSTREAMS);
  char *h_indata = (char *)malloc(sizeof(char)*16*NANT*96*8*2);
  float *bp = (float *)malloc(sizeof(float)*256);
  unsigned char *tmp_buf = (unsigned char *)malloc(sizeof(unsigned char)*256*48*4*NSTREAMS);  
  
  // streams and device  
  cudaStream_t stream[NSTREAMS];
  for (int st=0;st<NSTREAMS;st++) {
    cudaStreamCreate(&stream[st]);
    cudaMalloc((void **)&d_indata[st], 16*96*NANT*8*2*sizeof(char)); // data input to bf kernel
    cudaMalloc((void **)&d_outdata[st], 256*48*4*sizeof(unsigned char)); // data output from adder
    cudaMalloc((void **)&d_transfer[st], 256*96*16*sizeof(float)); // output from beamformer
    cudaMalloc((void **)&d_inr[st], 16*48*2*64*16*sizeof(half)); // real data
    cudaMalloc((void **)&d_ini[st], 16*48*2*64*16*sizeof(half)); // real data
    thrust::device_ptr<half> d1(d_inr[st]);
    thrust::fill(d1, d1+16*48*2*64*16, 0.0);
    thrust::device_ptr<half> d2(d_ini[st]);
    thrust::fill(d2, d2+16*48*2*64*16, 0.0);
  }

  
  
  // set up

  int observation_complete=0;
  int blocks = 0, started = 0;
  
  syslog(LOG_INFO, "starting observation");

  while (!observation_complete) {

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    // DO STUFF

    // calc weights
    init_weights(fnam,antpos,weights,flagants);
    cudaMemcpy(d_antpos, antpos, 64*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, 64*NW*2*2*sizeof(float), cudaMemcpyHostToDevice);  
    calc_weights<<<6144, 256>>>(d_antpos, d_weights, d_freqs, d_wr, d_wi);
    if (DEBUG) syslog(LOG_INFO,"Finished with weights");
    
    if (started==1) {

      // loop over ints
      for (int bst=0;bst<nints/NSTREAMS;bst++) {

	for (int st=0;st<NSTREAMS;st++) {


	  
	  // copy to h_indata
	  //memcpy(h_indata,block+(bst*NSTREAMS+st)*nbytes_per_int,nbytes_per_int);

	  // rotate h_indata in place
	  //reorder_block(h_indata);
	  
	  // copy to device
	  //cudaMemcpyAsync(d_indata, h_indata, 24576*NANT*sizeof(char), cudaMemcpyHostToDevice, stream[st]);
	  cudaMemcpyAsync(d_indata[st], block+(bst*NSTREAMS+st)*nbytes_per_int, 24576*NANT*sizeof(char), cudaMemcpyHostToDevice, stream[st]);

	  // do promotion
	  promoter<<<16*48*NANT, 32, 0, stream[st]>>>(d_indata[st], d_inr[st], d_ini[st]);
	  
	  // run beamformer kernel
	  beamformer<<<24576, 32, 0, stream[st]>>>(d_inr[st], d_ini[st], d_wr, d_wi, d_transfer[st], stuffants);
	  	  
	  // run adder kernel
	  adder<<<12288, 32, 0, stream[st]>>>(d_transfer[st], d_outdata[st], d_bp);
	  
	  // copy to host
	  cudaMemcpyAsync(tmp_buf + 256*48*4*st, d_outdata[st], 256*48*4*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[st]);
	  for (int j=0;j<12288*4;j++)
	    output_buffer[(bst*NSTREAMS+st)*12288*4+j] = tmp_buf[j+256*48*4*st];

	  if (DEBUG && bst*NSTREAMS+st==10) {
	    for (int j=0;j<48;j++) syslog(LOG_DEBUG,"%hu",output_buffer[(bst*NSTREAMS+st)*12288+BEAM_OUT*48+j]);
	  }        
	  
	}
      }


    }
    
    if (started==0) {
      syslog(LOG_INFO,"now in RUN state");
      started=1;

      // calculate bandpass

      for (int i=0;i<256;i++) bp[i] = 0.;
      
      // do standard bf but calculate bandpass

      // loop over ints
      for (int bst=0;bst<nints/NSTREAMS;bst++) {

	for (int st=0;st<NSTREAMS;st++) {
	  
	  // copy to h_indata
	  //memcpy(h_indata,block+(bst*NSTREAMS+st)*nbytes_per_int,nbytes_per_int);

	  // rotate h_indata in place - this is current
	  //reorder_block(h_indata);

	  // copy to device
	  //cudaMemcpyAsync(d_indata, h_indata, 24576*NANT*sizeof(char), cudaMemcpyHostToDevice, stream[st]);
	  cudaMemcpyAsync(d_indata[st], block+(bst*NSTREAMS+st)*nbytes_per_int, 24576*NANT*sizeof(char), cudaMemcpyHostToDevice, stream[st]);

	  // do promotion
	  promoter<<<16*48*NANT, 32, 0, stream[st]>>>(d_indata[st], d_inr[st], d_ini[st]);

	  //if (bst==0 && st==0) 
	  //  printer<<<3072, 32>>>(d_inr,d_ini);	  
	  
	  // run beamformer kernel
	  beamformer<<<24576, 32, 0, stream[st]>>>(d_inr[st], d_ini[st], d_wr, d_wi, d_transfer[st], stuffants);
	  
	  // copy back to host
	  cudaMemcpyAsync(h_transfer + st*256*96*16, d_transfer[st], sizeof(float)*393216, cudaMemcpyDeviceToHost, stream[st]);	

	  // calculate bandpass
	  //if (st==0 && bst==0) 
	  //calc_bp(h_transfer,bp,1);
	  calc_bp(h_transfer + st*256*96*16,bp,0);

	}
      }

      // adjust bandpass
      syslog(LOG_INFO,"Final BP...");
      for (int i=0;i<256;i++) {
	syslog(LOG_INFO,"coeff %d %g",i,bp[i]);
	if (bp[i]!=0.) {
	  bp[i] /= 48.*nints; 
	  bp[i] = 128./bp[i]/4.;
	}
      }
      cudaMemcpy(d_bp, bp, sizeof(float)*256, cudaMemcpyHostToDevice);
      
      // junk into output
      memset(output_buffer,0,block_out);
      
    }

    // write output for debug
    
    // write to output
    written = ipcio_write (hdu_out->data_block, (char *)(output_buffer), block_out);
    if (written < block_out)
      {
	syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	return EXIT_FAILURE;
      }

    if (DEBUG) {
      syslog(LOG_DEBUG, "written block %d",blocks);      
    }
    blocks++;
    

    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  for (int st=0;st<NSTREAMS;st++) {
    cudaStreamDestroy(stream[st]);
    cudaFree(d_indata[st]);
    cudaFree(d_outdata[st]);
    cudaFree(d_transfer[st]);
    cudaFree(d_inr[st]);
    cudaFree(d_ini[st]);
  }
  free(fnam);
  free(flagants);
  free(h_indata);
  free(output_buffer);
  free(antpos);
  free(weights);
  free(freqs);
  free(bp);
  free(h_transfer);
  free(tmp_buf);
  cudaFree(d_wr);
  cudaFree(d_wi);
  cudaFree(d_antpos);
  cudaFree(d_freqs);
  cudaFree(d_weights);
  cudaFree(d_wr);
  cudaFree(d_wi);
  cudaFree(d_bp);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


