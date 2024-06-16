#define __USE_GNU
#define _GNU_SOURCE
#include <sched.h>
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

#define NTIMES_P 4096	// # of time samples (assuming 1ms sampling period)
#define NCHAN_P 1024	// # of channels on BF node side
#define NBEAMS_P 64	// # of beams on BF side
#define M_P NTIMES_P
#define N_P 32
#define HDR_SIZE 4096
#define BUF_SIZE NTIMES_P*NCHAN_P*NBEAMS_P // size of TCP packet

// global variables
int DEBUG = 0;
double skarray[NBEAMS_P*NCHAN_P+1];	// array with SK values -- size NCHANS * NBEAMS
double avgspec[NBEAMS_P*NCHAN_P+1];	// spectrum over all beams to estimate median filter
double baselinecorrec[NBEAMS_P*NCHAN_P+1];	// spectrum over all beams to estimate median filter
int cores[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25};

void swap(char *p,char *q) {
   char t;
   
   t=*p; 
   *p=*q; 
   *q=t;
}

double medval(double a[],int n) { 
	int i,j;
	char tmp[n];
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

/* THREAD FUNCTION */

struct data {
	unsigned char * indata;
	double * inSK;
  unsigned char * output;
  int cnt;
	double nThreshUp;
	int n_threads;
	int thread_id;
	int debug;
};

void noise_inject(void *args) {
	
	struct data *d = args;
	int thread_id = d->thread_id;
	int dbg = d->debug;
	// set affinity
	const pthread_t pid = pthread_self();
	const int core_id = cores[thread_id];
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core_id, &cpuset);
	const int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
	if (set_result != 0)
		syslog(LOG_ERR,"thread %d: setaffinity_np fail",thread_id);
	const int get_affinity = pthread_getaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
	if (get_affinity != 0) 
		syslog(LOG_ERR,"thread %d: getaffinity_np fail",thread_id);
	if (CPU_ISSET(core_id, &cpuset))
	  if (dbg) syslog(LOG_DEBUG,"thread %d: successfully set thread",thread_id);
	
	
	// noise injection
	
	unsigned char *indata = (unsigned char *)d->indata;
	double *inSK = (double *)d->inSK;
	unsigned char *output = (unsigned char *)d->output;
	int * cnt = (int *)d->cnt;
	double nThreshUp = (double)d->nThreshUp;
	int nthreads = d->n_threads;
	int i, j, k;
	
	// copy from input to output
	//memcpy(output,indata,(NBEAMS_P/nthreads)*NTIMES_P*NCHAN_P);
	
	//cnt[thread_id] = 0;
	
	for (i = 0; i < (int)(NBEAMS_P/nthreads); i++){
	  for (k = 0; k < NCHAN_P; k++){
	    if (inSK[i*(int)(NCHAN_P) + k] > nThreshUp){
	      cnt[thread_id]++;
	      //if (dbg) syslog(LOG_DEBUG,"thread %d: flagging %d %d: sk %g",thread_id,i,k,inSK[i*(int)(NCHAN_P) + k]);
	      //for (j = 0; j < NTIMES_P; j++){
		//output[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = (unsigned char)(20. * rand() / ( (double)RAND_MAX ) + 10.);
		//indata[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = (unsigned char)(20. * 1. / ( (double)RAND_MAX ) + 10.);
	      //}

	      // copy from lookup table
	      for (j = 0; j < NTIMES_P; j++)
		indata[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = output[k*NTIMES_P+j];
	      
	    }
	    /*else{
	      for (j = 0; j < NTIMES_P; j++){
	      output[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = indata[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k];
	      }
	      }*/
	  }
	}
	
	
	
	if (dbg) syslog(LOG_DEBUG,"thread %d: done - freeing",thread_id);
	int thread_result = 0;
	pthread_exit((void *) &thread_result);
}

/* END THREAD FUNCTION */

void usage()
{
  fprintf (stdout,
	   "flagger [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default dada]\n"
	   " -o out_key [default caca]\n"
	   " -n use noise generation rather than zeros\n"
	   " -t SK threshold [default 5.0]\n"
	   " -b compute and apply baseline correction\n"
	   " -h print usage\n");
}


int main(int argc, char**argv)
{

  // syslog start
  openlog ("flagger", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  // threads initialization
  int nthreads = 16;
  pthread_t threads[nthreads];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;
  
  // read command line args

  // data block HDU keys
  key_t in_key = 0x0000dada;
  key_t out_key = 0x0000caca;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  int noise = 0;
  double skthresh = 5.0;
  int bcorr = 0;
  
  while ((arg=getopt(argc,argv,"c:t:i:o:bndh")) != -1)
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
	case 't':
	  if (optarg)
	    {
	      skthresh = atof(optarg);
	      syslog(LOG_INFO,"modified SKTHRESH to %g",skthresh);
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
	case 'b':
	  bcorr=1;
	  syslog (LOG_INFO, "Will calculate and apply baseline correction");
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
	
  /* //read fake header for now
	char head_dada[4096];
	FILE *f = fopen("/home/dsa/dsa110-xengine/src/correlator_header_dsaX.txt", "rb");
	fread(head_dada, sizeof(char), 4096, f);
	fclose(f); */
  
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
  
  ////////////////		
	
  double S1 = 0;
  double S2 = 0;
  double sampval;
  double nThreshUp = skthresh;	// Threshold to apply to SK (empirical estimation)
  struct data args[16];
  int * flag_counts = (int *)malloc(sizeof(int)*nthreads);
  //unsigned char * output = (unsigned char *)malloc(sizeof(char)*NBEAMS_P*NCHAN_P*NTIMES_P);
  int nFiltSize = 21;
  int cnt = 0;

  // make array of random numbers
  unsigned char * lookup_rand = (unsigned char *)malloc(sizeof(unsigned char)*NTIMES_P*NCHAN_P);
  for (int i=0;i<NTIMES_P*NCHAN_P;i++) 
    lookup_rand[i] = (unsigned char)(20. * rand() / ( (double)RAND_MAX ) + 10.);
  
  // put rest of the code inside while loop
  while (1) {	
    
    // read a DADA block
    cin_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    in_data = (unsigned char *)(cin_data);
    
    // compute SK and averaged spectrum
    S1 = 0;
    S2 = 0;
    sampval = 0;
		
    for (int i = 0; i < NBEAMS_P; i++){
      for (int k = 0; k < NCHAN_P; k++){
	for (int j = 0; j < NTIMES_P; j++){
	  sampval = (double)in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k];
	  avgspec[i*(int)(NCHAN_P) + k] += sampval / NTIMES_P;
	  S1 += sampval;
	  S2 += sampval * sampval;
	  skarray[i*(int)(NCHAN_P) + k] = (double)((M_P*N_P+1) / (M_P-1) * ( (M_P*S2)/(S1*S1) - 1 ));
	}
	S1 = 0;
	S2 = 0;
      }
    }
    if (DEBUG) syslog (LOG_DEBUG,"has computed SK.");
    if (DEBUG) syslog(LOG_DEBUG,"example SK value : %g", (double)skarray[10]);
		
    // compute baseline correction
    if (bcorr) {
      for (int i = 0; i < NBEAMS_P*NCHAN_P-nFiltSize; i++)
	baselinecorrec[i] = medval(&avgspec[i],nFiltSize);
    }
    		
    
    // compare SK values to threshold and
    // replace thresholded channels with noise or 0
    
    if (noise){

      for (int i=0;i<nthreads;i++) flag_counts[i] = 0;
      for (int i=0; i<nthreads; i++) {
	args[i].indata = in_data + i*(int)((NBEAMS_P/nthreads)*NCHAN_P*NTIMES_P);
	args[i].inSK = skarray + i*(int)(NBEAMS_P/nthreads*NCHAN_P);
	args[i].output = lookup_rand;
	args[i].cnt = flag_counts;
	args[i].nThreshUp = nThreshUp;
	args[i].n_threads = nthreads;
	args[i].thread_id = i;
	args[i].debug = DEBUG;
      }
      if (DEBUG) syslog(LOG_DEBUG,"creating %d threads",nthreads);
      for(int i=0; i<nthreads; i++){
	if (pthread_create(&threads[i], &attr, &noise_inject, (void *)(&args[i]))) {
	  syslog(LOG_ERR,"Failed to create noise_inject thread %d\n", i);
	}
      }
      /*for(int i=0; i<nthreads; i++){
	for(int j=0; j<(int)(NBEAMS_P/nthreads*NCHAN_P*NTIMES_P); i++){
	  in_data[i*(int)(NBEAMS_P/nthreads*NCHAN_P*NTIMES_P)+j] = args[i].output[j];
	}
	}*/
      pthread_attr_destroy(&attr);

      for(int i=0; i<nthreads; i++){
	pthread_join(threads[i], &result);
	if (DEBUG) syslog(LOG_DEBUG,"joined thread %d",i);
      }

      cnt = 0;
      for(int i=0; i<nthreads; i++) cnt += flag_counts[i];
      //memcpy(in_data,output,sizeof(in_data));
    }
    else{
      for (int i = 0; i < NBEAMS_P; i++){
	for (int k = 0; k < NCHAN_P; k++){
	  if (skarray[i*(int)(NCHAN_P) + k] > nThreshUp){
	    cnt++;
	    for (int j = 0; j < NTIMES_P; j++){
	      in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = 0;
	    }
	  }
	}
      }
    }
    syslog (LOG_INFO,"%d channels*baselines flagged",cnt);
		
    // apply baseline correction
    if (bcorr) {
      for (int i = 0; i < NBEAMS_P; i++){
	for (int k = 0; k < NCHAN_P; k++){
	  for (int j = 0; j < NTIMES_P; j++){
	    //in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = (unsigned char)(in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] / (unsigned char)baselinecorrec[i*(int)NCHAN_P+k]);
	    in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = (unsigned char)((double)(in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k]) / baselinecorrec[i*(int)NCHAN_P+k]);
	  }
	}
      }
      
      syslog (LOG_DEBUG,"baseline correction applied");
    }
		
    // close block after reading
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    if (DEBUG) syslog(LOG_DEBUG,"closed read block");		
    
    written = ipcio_write (hdu_out->data_block, (char *)(in_data), BUF_SIZE);
    if (written < BUF_SIZE)
      {
	syslog(LOG_ERR,"write error");
	return EXIT_FAILURE;
      }

    if (DEBUG) syslog (LOG_DEBUG,"write flagged data done.");
		
    
  }

  free(lookup_rand);
  return 0;    
} 
