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
  for (int i=idx0;i<idx0+NCHAN_P*(NTIMES_P/NTHREADS_GPU);i+=NCHAN_P)
    csum[thread_id] += (float)(data[i]);

  __syncthreads();
  
  // sum into shared memory
  int maxn = NTHREADS_GPU/2;
  int act_maxn = maxn;
  if (thread_id<maxn) {
    while (act_maxn>0) {
      csum[thread_id] += csum[thread_id+act_maxn];
      act_maxn = (int)(act_maxn/2);
    }
  }

  if (thread_id==0) {    
    spectrum[bm*NCHAN_P+ch] = csum[thread_id] / (1.*NTIMES_P);
  }

}

// kernel to do flagging
// launch with NBEAMS_P*NCHAN_P blocks of NTHREADS_GPU threads 
__global__
void flag(unsigned char *data, int * mask, unsigned char * repval) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  int bm =(int)( block_id/NCHAN_P);
  int ch = (int)(block_id % (NCHAN_P));
  int tm0 = (int)(thread_id*(NTIMES_P/NTHREADS_GPU));
  
  // do replacement
  int mask_idx = bm*NCHAN_P + ch;
  int idx0 = bm*NTIMES_P*NCHAN_P + tm0*NCHAN_P + ch;
  int ridx = ch*NTIMES_P + tm0;
  if (mask[mask_idx]==1) {
    for (int i=idx0;i<idx0+NCHAN_P*(NTIMES_P/NTHREADS_GPU);i+=NCHAN_P) {
      data[i] = repval[ridx];
      ridx++;
    }
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
  memset(mask, 0, NBEAMS_P*NCHAN_P*sizeof(int));
  for (i = 0; i < NBEAMS_P; i++){    
    for (j = ZeroChannels; j < NCHAN_P-ZeroChannels; j++){
      if (CorrecSpec[i*NCHAN_P+j] > Thr * madspec[i] || CorrecSpec[i*NCHAN_P+j] < - Thr * madspec[i])
	mask[i*NCHAN_P+j] = 1;
    }
    
  }
  
  for (i=0;i<NCHAN_P;i++)
    printf("%g %g %g\n",CorrecSpec[i],madspec[0]*Thr,spec[i]);

  free(baselinecorrec);
  free(CorrecSpec);
  free(medspec);
  free(madspec);
  free(normspec);
  
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
  char * fnam;
  FILE *fout;
  fnam = (char *)malloc(sizeof(char)*200);
  int fwrite = 0;
  
  while ((arg=getopt(argc,argv,"c:t:i:o:f:ndh")) != -1)
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

  // make array of random numbers
  unsigned char * lookup_rand = (unsigned char *)malloc(sizeof(unsigned char)*NTIMES_P*NCHAN_P);
  for (int i=0;i<NTIMES_P*NCHAN_P;i++) {
    if (noise) 
      lookup_rand[i] = (unsigned char)(20. * rand() / ( (double)RAND_MAX ) + 10.);
    else
      lookup_rand[i] = 0;
  }

  if (DEBUG) syslog(LOG_INFO,"finished with lookup table");

  // declare stuff for host and GPU
  unsigned char * d_data;
  cudaMalloc((void **)&d_data, NBEAMS_P*NTIMES_P*NCHAN_P*sizeof(unsigned char));
  unsigned char * h_data = (unsigned char *)malloc(sizeof(unsigned char)*NBEAMS_P*NTIMES_P*NCHAN_P);
  int * h_mask = (int *)malloc(sizeof(int)*NBEAMS_P*NCHAN_P);
  int * d_mask;
  cudaMalloc((void **)&d_mask, NBEAMS_P*NCHAN_P*sizeof(int));
  float * d_spec;
  cudaMalloc((void **)&d_spec, NBEAMS_P*NCHAN_P*sizeof(float));
  float * h_spec = (float *)malloc(sizeof(float)*NBEAMS_P*NCHAN_P);
  unsigned char * d_repval;
  cudaMalloc((void **)&d_repval, NTIMES_P*NCHAN_P*sizeof(unsigned char));
  cudaMemcpy(d_repval, lookup_rand, NTIMES_P*NCHAN_P*sizeof(unsigned char), cudaMemcpyHostToDevice);
  
  // put rest of the code inside while loop
  while (1) {	
    
    // read a DADA block
    cin_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    in_data = (unsigned char *)(cin_data);

    if (DEBUG) syslog(LOG_INFO,"read block");
    
    // compute the mean spectrum
    if (DEBUG) syslog(LOG_INFO,"starting spectrum calc");
    cudaMemcpy(d_data, in_data, NBEAMS_P*NTIMES_P*NCHAN_P*sizeof(unsigned char), cudaMemcpyHostToDevice);
    calc_spectrum<<<NBEAMS_P*NCHAN_P, NTHREADS_GPU>>>(d_data, d_spec);
    cudaMemcpy(h_spec, d_spec, NBEAMS_P*NCHAN_P*sizeof(float), cudaMemcpyDeviceToHost);
    if (DEBUG) syslog(LOG_INFO,"finished spectrum calc");

    // figure out flagging - fill h_mask
    channflag(h_spec, thresh, h_mask);
    
    // do flagging
    if (DEBUG) syslog(LOG_INFO,"starting flagging calc");
    cudaMemcpy(d_mask, h_mask, NBEAMS_P*NCHAN_P*sizeof(int), cudaMemcpyHostToDevice);
    flag<<<NBEAMS_P*NCHAN_P, NTHREADS_GPU>>>(d_data, d_mask, d_repval);
    cudaMemcpy(h_data, d_data, NBEAMS_P*NTIMES_P*NCHAN_P*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (DEBUG) syslog(LOG_INFO,"finished flagging calc");
		
    // close block after reading
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    if (DEBUG) syslog(LOG_DEBUG,"closed read block");		
    
    written = ipcio_write (hdu_out->data_block, (char *)(h_data), BUF_SIZE);
    if (written < BUF_SIZE)
      {
	syslog(LOG_ERR,"write error");
	return EXIT_FAILURE;
      }

    if (DEBUG) syslog (LOG_INFO,"write flagged data done.");
    if (fwrite) {
      fout=fopen(fnam,"a");
      for (int i=0;i<NCHAN_P;i++) fprintf(fout,"%d %g\n",h_mask[i],h_spec[i]);
      fclose(fout);
    }
     
    for (int i=0;i<NBEAMS_P*NCHAN_P;i++)
      h_mask[0] += h_mask[i];
    syslog(LOG_INFO,"FLAG_COUNT %d",h_mask[0]);   		
    
  }

  free(lookup_rand);
  free(fnam);
  free(h_data);
  free(h_mask);
  free(h_spec);
  cudaFree(d_data);
  cudaFree(d_spec);
  cudaFree(d_mask);
  cudaFree(d_repval);
  return 0;    
} 
