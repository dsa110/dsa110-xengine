// -*- c++ -*-
/*#include <sched.h>
#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <syslog.h>


#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_capture.h"
*/
#include <iostream>
#include <algorithm>
using std::cout;
using std::cerr;
using std::endl;
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/syscall.h>
#include <syslog.h>
#include <curand.h>
#include <curand_kernel.h>

#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"

#include <src/sigproc.h>
#include <src/header.h>


#define NTIMES_P 4096	// # of time samples (assuming 1ms sampling period)
#define NCHAN_P 1024	// # of channels on BF node side
#define NBEAMS_P 64	// # of beams on BF side
#define M_P NTIMES_P
#define N_P 32
#define HDR_SIZE 4096
#define BUF_SIZE NTIMES_P*NCHAN_P*NBEAMS_P // size of TCP packet
#define NTHREADS_GPU 32
#define MN 64.0
#define SIG 8.0
#define RMAX 16384

// global variables
int DEBUG = 0;

// kernel to calculate mean spectrum
// launch with NBEAMS_P*NCHAN_P blocks of NTHREADS_GPU threads 
__global__
void calc_spectrum(unsigned char *data, float * spectrum) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  __shared__ float csum[NTHREADS_GPU];
  csum[thread_id] = 0.;

  int bm =(int)( block_id/NCHAN_P);
  int ch = (int)(block_id % (NCHAN_P));
  int tm0 = (int)(thread_id*(NTIMES_P/NTHREADS_GPU));
  
  // find sum of local times
  int idx0 = bm*NTIMES_P*NCHAN_P + tm0*NCHAN_P + ch;
  for (int tm=0; tm<NTIMES_P/NTHREADS_GPU; tm++) {    
    csum[thread_id] += (float)(data[idx0]);
    idx0 += NCHAN_P;
  }

  __syncthreads();
  
  // sum into shared memory
  if (thread_id<16) {
    csum[thread_id] += csum[thread_id+16];
    __syncthreads();
    csum[thread_id] += csum[thread_id+8];
      __syncthreads();
    csum[thread_id] += csum[thread_id+4];
      __syncthreads();
    csum[thread_id] += csum[thread_id+2];
      __syncthreads();
    csum[thread_id] += csum[thread_id+1];
      __syncthreads();
  }
  /*  
  int maxn = NTHREADS_GPU/2;
  int act_maxn = maxn;
  if (thread_id<maxn) {
    while (act_maxn>0) {
      csum[thread_id] += csum[thread_id+act_maxn];
      act_maxn = (int)(act_maxn/2);
    }
  }
  */
  
  if (thread_id==0) {    
    spectrum[bm*NCHAN_P+ch] = csum[thread_id] / (1.*NTIMES_P);
  }

}


// kernel to calculate variance spectrum
// launch with NBEAMS_P*NCHAN_P blocks of NTHREADS_GPU threads 
__global__
void calc_varspec(unsigned char *data, float * spectrum, float * varspec) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  __shared__ float csum[NTHREADS_GPU];
  csum[thread_id] = 0.;

  int bm =(int)( block_id/NCHAN_P);
  int ch = (int)(block_id % (NCHAN_P));
  int tm0 = (int)(thread_id*(NTIMES_P/NTHREADS_GPU));
  float val;
  
  // find sum of local times
  int idx0 = bm*NTIMES_P*NCHAN_P + tm0*NCHAN_P + ch;
  for (int tm=0; tm<NTIMES_P/NTHREADS_GPU; tm++) {    
    val = (float)(data[idx0]) - spectrum[bm*NCHAN_P + ch];
    csum[thread_id] += val*val;
    idx0 += NCHAN_P;
  }
  
  __syncthreads();
  
  // sum into shared memory
  if (thread_id<16) {
    csum[thread_id] += csum[thread_id+16];
    __syncthreads();
    csum[thread_id] += csum[thread_id+8];
        __syncthreads();
    csum[thread_id] += csum[thread_id+4];
        __syncthreads();
    csum[thread_id] += csum[thread_id+2];
        __syncthreads();
    csum[thread_id] += csum[thread_id+1];
        __syncthreads();
  }
  /*
  int maxn = NTHREADS_GPU/2;
  int act_maxn = maxn;
  if (thread_id<maxn) {
    while (act_maxn>0) {
      csum[thread_id] += csum[thread_id+act_maxn];
      act_maxn = (int)(act_maxn/2);
    }
    }*/

  if (thread_id==0) {    
    varspec[bm*NCHAN_P+ch] = csum[thread_id] / (1.*NTIMES_P);
  }

}

// kernel to calculate maximum value
// launch with NBEAMS_P*NCHAN_P blocks of NTHREADS_GPU threads 
__global__
void calc_maxspec(unsigned char *data, float * maxspec) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  __shared__ float csum[NTHREADS_GPU];
  csum[thread_id] = 0.;

  int bm =(int)( block_id/NCHAN_P);
  int ch = (int)(block_id % (NCHAN_P));
  int tm0 = (int)(thread_id*(NTIMES_P/NTHREADS_GPU));
  float val=0.;
  
  // find max of local times
  int idx0 = bm*NTIMES_P*NCHAN_P + tm0*NCHAN_P + ch;
  for (int i=idx0;i<idx0+NCHAN_P*(NTIMES_P/NTHREADS_GPU);i+=NCHAN_P) {
    if ((float)(data[i])>val) val = (float)(data[i]);
  }
  csum[thread_id] = val;
  
  __syncthreads();
  
  // sum into shared memory
  int maxn = NTHREADS_GPU/2;
  int act_maxn = maxn;
  if (thread_id<maxn) {
    while (act_maxn>0) {
      if (csum[thread_id]<csum[thread_id+act_maxn])
	csum[thread_id]=csum[thread_id+act_maxn];
      act_maxn = (int)(act_maxn/2);
    }
  }

  if (thread_id==0) {    
    maxspec[bm*NCHAN_P+ch] = csum[thread_id];
  }

}

// kernel to scale data
// launch with NBEAMS_P*NTIMES_P*NCHAN_P/NTHREADS_GPU blocks of NTHREADS_GPU threads
__global__
void scaley(unsigned char *data, float *spectrum, float *varspec) {

  int idx = blockIdx.x*NTHREADS_GPU + threadIdx.x;
  int bm = (int)(idx / (NTIMES_P*NCHAN_P));
  int ch = (int)(idx % NCHAN_P);
  int spidx = bm*NCHAN_P+ch;

  float val = (float)(data[idx]);
  val = (val-spectrum[spidx])*(SIG/sqrtf(varspec[spidx])) + MN;
  data[idx] = (unsigned char)((__float2uint_rn(2.*val))/2);
  

}

// kernel to do flagging
// launch with n_mask*NTIMES_P/NTHREADS_GPU blocks of NTHREADS_GPU threads 
__global__
void flag(unsigned char *data, int * midx, unsigned char *repval) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int midx_idx = (int)(block_id/(NTIMES_P/NTHREADS_GPU));
  
  int bm = (int)(midx[midx_idx] / NCHAN_P);
  int ch = (int)(midx[midx_idx] % NCHAN_P);
  int tm = ((int)(block_id % (NTIMES_P/NTHREADS_GPU)))*NTHREADS_GPU + thread_id;
  int idx = bm*NTIMES_P*NCHAN_P + tm*NCHAN_P + ch;  

  // do replacement
  data[idx] = repval[ch*NTIMES_P+tm];
    
}

// kernel to make random numbers
// launch with NTIMES_P*NCHAN_P/NTHREADS_GPU blocks of NTHREADS_GPU threads 
__global__
void genrand(unsigned char *repval, unsigned int seed) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  
  // for random number
  curandState_t state;
  float u1, u2, va;
  curand_init(seed, block_id*NTHREADS_GPU+thread_id, 1, &state);
  u1 = ((float)(curand(&state) % RMAX))/(1.*RMAX);
  u2 = ((float)(curand(&state) % RMAX))/(1.*RMAX);
  va = sqrtf(-2.*logf(u1))*cosf(2.*M_PI*u2);

  // do replacement
  repval[block_id*NTHREADS_GPU+thread_id] = (unsigned char)(__float2uint_rn(2.*(va*SIG+MN))/2);
    
}



// assumed spec has size NBEAMS_P*NCHAN_P
// ref is reference value
void genmask(float *spec, float thresh, float ref, int *mask) {

  for (int i=0;i<NBEAMS_P*NCHAN_P;i++) {
    if (fabs(spec[i]-ref)>thresh) mask[i] = 1;
  }

}


void swap(float *p,float *q) {
   float t;
   
   t=*p; 
   *p=*q; 
   *q=t;
}

float medval(float *a,int n) { 
	int i,j;
	float tmp[n];
	for (i = 0;i < n;i++)
		tmp[i] = a[i];
	
	for(i = 0;i < n-1;i++) {
		for(j = 0;j < n-i-1;j++) {
			if(tmp[j] > tmp[j+1])
				swap(&tmp[j],&tmp[j+1]);
		}
	}
	return tmp[(n+1)/2-1];
}

void channflag(float* spec, float Thr, int * mask);

void channflag(float* spec, float Thr, int * mask) {
	
  int i, j;
  float* baselinecorrec;	// baseline correction
  float* CorrecSpec;			// corrected spectrum
  float* medspec;			// median values for each beam spectrum
  float* madspec;			// mad for each beam spectrum
  float* normspec;			// corrected spec - median value (for MAD calculation)

  baselinecorrec = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  CorrecSpec = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  medspec = (float *)malloc(sizeof(float)*NBEAMS_P);
  madspec = (float *)malloc(sizeof(float)*NBEAMS_P);
  normspec = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  
  
  int ZeroChannels = 128; 
  int nFiltSize = 21;
  
  // calculate median filtered spectrum and correct spectrum at the same time
  for (i = 0; i < NBEAMS_P*NCHAN_P-nFiltSize; i++){
    baselinecorrec[i] = medval(&spec[i],nFiltSize);
    CorrecSpec[i] = spec[i] - baselinecorrec[i];
  }
	
  // calculate median value for each beam
  for (i = 0; i < NBEAMS_P; i++)
    medspec[i] = medval(&CorrecSpec[i*NCHAN_P],NCHAN_P);
  
  // compute MAD for each beam
  for (i = 0; i < NBEAMS_P; i++){
    for (j = ZeroChannels; j < NCHAN_P-ZeroChannels; j++){
      normspec[j-ZeroChannels] = abs(CorrecSpec[j]-medspec[i]);
    }
    madspec[i] = medval(normspec,NCHAN_P-2*ZeroChannels);
  }
	
  // mask  
  for (i = 0; i < NBEAMS_P; i++){    
    for (j = ZeroChannels; j < NCHAN_P-ZeroChannels; j++){
      if (CorrecSpec[i*NCHAN_P+j] > Thr * madspec[i] || CorrecSpec[i*NCHAN_P+j] < - Thr * madspec[i])
	mask[i*NCHAN_P+j] = 1;
    }
    
  }
  
  //for (i=0;i<NCHAN_P;i++)
  //  printf("%g %g %g\n",CorrecSpec[i],madspec[0]*Thr,spec[i]);

  free(baselinecorrec);
  free(CorrecSpec);
  free(medspec);
  free(madspec);
  free(normspec);
  
}

// to gather mask indices
void gather_mask(int *h_idx, int *h_mask, int *n_mask) {

  (*n_mask) = 0;
  for (int i=0;i<NBEAMS_P*NCHAN_P;i++) {
    if (h_mask[i]==1) {      
      h_idx[(*n_mask)] = i;
      if (DEBUG) syslog(LOG_INFO,"%d %d %d",i,h_mask[i],(*n_mask));
      (*n_mask) += 1;
    }
  }

}


void usage()
{
  fprintf (stdout,
	   "flagger [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default dada]\n"
	   " -o out_key [default caca]\n"
	   " -n use noise generation rather than zeros\n"
	   " -t flagging threshold [default 5.0]\n"
	   " -v variance flagging\n"
	   " -h print usage\n");
}


int main(int argc, char**argv)
{

  // syslog start
  multilog_t* log = 0;
  openlog ("gpu_flagger", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  // set cuda device
  cudaSetDevice(1);
  
  // read command line args

  // data block HDU keys
  key_t in_key = 0x0000dada;
  key_t out_key = 0x0000caca;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  int noise = 0;
  double thresh = 5.0;
  int varf = 0;
  char * fnam;
  FILE *fout;
  fnam = (char *)malloc(sizeof(char)*200);
  int fwrite = 0;
  
  while ((arg=getopt(argc,argv,"c:t:i:o:f:vndh")) != -1)
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
	case 'f':
	  if (optarg)
	    {
	      strcpy(fnam,optarg);
	      fwrite = 1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
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
	case 't':
	  if (optarg)
	    {
	      thresh = atof(optarg);
	      syslog(LOG_INFO,"modified THRESH to %g",thresh);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-t flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }

	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'n':
	  noise=1;
	  syslog (LOG_INFO, "Will generate noise samples");
	  break;	  
	case 'v':
	  varf=1;
	  syslog (LOG_INFO, "Will do variance flagging");
	  break;	  
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // Bind to cpu core
  if (core >= 0)
    {
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
      syslog(LOG_NOTICE,"bound to core %d", core);
    }
  
  
  // CONNECT AND READ FROM BUFFER

  dada_hdu_t* hdu_in = 0;	// header and data unit
  uint64_t blocksize = NTIMES_P*NCHAN_P*NBEAMS_P;	// size of buffer
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to input buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to input buffer");
    return EXIT_FAILURE;
  }

  if (DEBUG) syslog(LOG_INFO,"connected to input buffer");
  
  uint64_t header_size = 0;
  // read the header from the input HDU
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  
  // mark the input header as cleared
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0){
    syslog (LOG_ERR,"could not mark header as cleared");
    return EXIT_FAILURE;
  }
  
  uint64_t block_id, bytes_read = 0;
  unsigned char *in_data;
  char *cin_data;
	     	
  // OUTPUT BUFFER
  dada_hdu_t* hdu_out = 0;
  hdu_out  = dada_hdu_create ();
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog (LOG_ERR,"flagged_data: could not connect to dada buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write (hdu_out) < 0) {
    syslog (LOG_ERR,"flagged_data: could not lock to dada buffer");
    return EXIT_FAILURE;
  }

  if (DEBUG) syslog(LOG_INFO,"connected to output");
  
  
  //// OUTPUT BUFFER
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  header_size = HDR_SIZE;
  if (!header_out)
    {
      syslog(LOG_ERR,"couldn't read header_out");
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      return EXIT_FAILURE;
    }
  uint64_t written=0;

  if (DEBUG) syslog(LOG_INFO,"copied header");
  
  ////////////////		

  // declare stuff for host and GPU
  unsigned char * d_data;
  cudaMalloc((void **)&d_data, NBEAMS_P*NTIMES_P*NCHAN_P*sizeof(unsigned char));
  unsigned char * h_data = (unsigned char *)malloc(sizeof(unsigned char)*NBEAMS_P*NTIMES_P*NCHAN_P);
  int * h_mask = (int *)malloc(sizeof(int)*NBEAMS_P*NCHAN_P);
  int * d_mask;
  cudaMalloc((void **)&d_mask, NBEAMS_P*NCHAN_P*sizeof(int));
  float * d_spec, * d_oldspec;
  cudaMalloc((void **)&d_spec, NBEAMS_P*NCHAN_P*sizeof(float));
  cudaMalloc((void **)&d_oldspec, NBEAMS_P*NCHAN_P*sizeof(float));
  float * h_spec = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  float * h_subspec = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  float * h_var = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  float * h_max = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  float * h_oldspec = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  float *d_spec0, *d_var0;
  cudaMalloc((void **)&d_spec0, NBEAMS_P*NCHAN_P*sizeof(float));
  cudaMalloc((void **)&d_var0, NBEAMS_P*NCHAN_P*sizeof(float));
  for (int i=0;i<NBEAMS_P*NCHAN_P;i++) h_oldspec[i] = 0.;
  cudaMemcpy(d_oldspec, h_oldspec, NBEAMS_P*NCHAN_P*sizeof(float), cudaMemcpyHostToDevice);
  float * d_var, * d_max;
  cudaMalloc((void **)&d_var, NBEAMS_P*NCHAN_P*sizeof(float));
  cudaMalloc((void **)&d_max, NBEAMS_P*NCHAN_P*sizeof(float));
  int * h_idx = (int *)malloc(sizeof(int)*NBEAMS_P*NCHAN_P);
  int * d_idx;
  cudaMalloc((void **)&d_idx, NBEAMS_P*NCHAN_P*sizeof(int));
  int n_mask = 0;

  // random numbers
  unsigned char *d_repval;
  cudaMalloc((void **)&d_repval, NTIMES_P*NCHAN_P*sizeof(unsigned char));
  genrand<<<NTIMES_P*NCHAN_P/NTHREADS_GPU,NTHREADS_GPU>>>(d_repval,time(NULL));
  syslog(LOG_INFO,"done with repvals");
  
  int started = 0;
  
  // put rest of the code inside while loop
  while (1) {	
    
    // read a DADA block
    cin_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    in_data = (unsigned char *)(cin_data);

    if (DEBUG) syslog(LOG_INFO,"read block");

    /* 
       if not first block, correct data
       1 - measure spectrum
       2 - measure varspec
       if first block, proceed.
       else
       3 - measure maximum value
       4 - use three spectra to derive channel flags
       5 - flag
     */

    // copy data to device
    cudaMemcpy(d_data, in_data, NBEAMS_P*NTIMES_P*NCHAN_P*sizeof(unsigned char), cudaMemcpyHostToDevice);
    //cudaMemset(d_data, 8, NBEAMS_P*NTIMES_P*NCHAN_P);

    // if not first block, correct data
    if (started==1) 
      scaley<<<NBEAMS_P*NTIMES_P*NCHAN_P/NTHREADS_GPU,NTHREADS_GPU>>>(d_data, d_spec0, d_var0);

    if (DEBUG) syslog(LOG_INFO,"copied data and scaled");
    
    // measure spectrum and varspec
    calc_spectrum<<<NBEAMS_P*NCHAN_P, NTHREADS_GPU>>>(d_data, d_spec);
    calc_varspec<<<NBEAMS_P*NCHAN_P, NTHREADS_GPU>>>(d_data, d_spec, d_var);
    cudaMemcpy(h_spec, d_spec, NBEAMS_P*NCHAN_P*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_var, d_var, NBEAMS_P*NCHAN_P*sizeof(float), cudaMemcpyDeviceToHost);

    if (DEBUG) syslog(LOG_INFO,"done spec and var");
    
    // if not first block
    if (started==1) {

      // calc maxspec
      calc_spectrum<<<NBEAMS_P*NCHAN_P, NTHREADS_GPU>>>(d_data, d_max);

      // derive channel flags
      cudaMemcpy(h_max, d_max, NBEAMS_P*NCHAN_P*sizeof(float), cudaMemcpyDeviceToHost);
      for (int i=0;i<NBEAMS_P*NCHAN_P;i++) {
	h_mask[i] = 0;
	h_subspec[i] = h_spec[i]-h_oldspec[i];
      }
      channflag(h_spec,thresh,h_mask);
      channflag(h_var,thresh,h_mask);
      channflag(h_max,thresh,h_mask);      

      // apply mask
      gather_mask(h_idx, h_mask, &n_mask);
      if (DEBUG) syslog(LOG_INFO,"FLAG_COUNT %d",n_mask);   		
      cudaMemcpy(d_idx, h_idx, n_mask*sizeof(int), cudaMemcpyHostToDevice);
      flag<<<n_mask*NTIMES_P/NTHREADS_GPU, NTHREADS_GPU>>>(d_data, d_idx, d_repval);      

      // write out stuff
      //for (int i=0;i<NBEAMS_P*NCHAN_P;i++)
      //	h_mask[0] += h_mask[i];
      //syslog(LOG_INFO,"FLAG_COUNT %d",h_mask[0]);   		

      
    }

    // copy data to host and write to buffer
    cudaMemcpy(h_data, d_data, NBEAMS_P*NTIMES_P*NCHAN_P*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // close block after reading
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    if (DEBUG) syslog(LOG_DEBUG,"closed read block");		    
    written = ipcio_write (hdu_out->data_block, (char *)(h_data), BUF_SIZE);
    if (written < BUF_SIZE)
      {
	syslog(LOG_ERR,"write error");
	return EXIT_FAILURE;
      }

    // deal with started and oldspec
    if (started==0) {
      cudaMemcpy(d_spec0, d_spec, NBEAMS_P*NCHAN_P*sizeof(float), cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_var0, d_var, NBEAMS_P*NCHAN_P*sizeof(float), cudaMemcpyDeviceToDevice);
      started=1;
    }
    for (int i=0;i<NBEAMS_P*NCHAN_P;i++) {
      h_oldspec[i] = h_spec[i];
    }
    
    if (fwrite) {
      fout=fopen(fnam,"a");
      for (int i=0;i<NCHAN_P;i++) fprintf(fout,"%d %g %g %g\n",h_mask[i],h_spec[i],h_var[i],h_max[i]);
      fclose(fout);
    }

    if (DEBUG) syslog(LOG_INFO,"done with round");
    
  }

  free(fnam);
  free(h_data);
  free(h_mask);
  free(h_spec);
  free(h_var);
  free(h_max);
  cudaFree(d_data);
  cudaFree(d_spec);
  cudaFree(d_var);
  cudaFree(d_mask);
  cudaFree(d_spec0);
  cudaFree(d_var0);
  return 0;    
} 
