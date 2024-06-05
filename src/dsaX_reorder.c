/* will reorder raw data for input to xgpu */
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
#include "dsaX_def.h"

#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>

// data to pass to threads
struct data {
  char * in;
  char * out;
  int n_threads;
  int thread_id;
  int debug;
};

/* global variables */
int DEBUG = 0;
int cores[16] = {4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

void dsaX_dbgpu_cleanup (dada_hdu_t * in, int write);
int dada_bind_thread_to_core (int core);

void dsaX_dbgpu_cleanup (dada_hdu_t * in, int write)
{

  if (write==0) {
  
    if (dada_hdu_unlock_read (in) < 0)
      {
	syslog(LOG_ERR, "could not unlock read on hdu_in");
      }
    dada_hdu_destroy (in);

  }

  if (write==1) {

    if (dada_hdu_unlock_write (in) < 0)
      {
	syslog(LOG_ERR, "could not unlock write on hdu_in");
      }
    dada_hdu_destroy (in);

  }
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_reorder_raw [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -t number of threads [default 4]\n"
	   " -i input key [default CAPTURED_BLOCK_KEY]\n"
	   " -o output key [default REORDER_BLOCK_KEY]\n"
	   " -q quitting after testing\n"
	   " -h print usage\n");
}

/* thread for data massaging */
void * massage(void *args) {

  // basic stuff
  struct data *d = args;
  int thread_id = d->thread_id;
  int dbg = d->debug;
   
  // masks for fluffing
  __m512i masks[4];
  masks[0] = _mm512_set_epi64(0x000f000f000f000fULL, 0x000f000f000f000fULL, 0x000f000f000f000fULL, 0x000f000f000f000fULL, 0x000f000f000f000fULL, 0x000f000f000f000fULL, 0x000f000f000f000fULL, 0x000f000f000f000fULL);
  masks[1] = _mm512_set_epi64(0x00f000f000f000f0ULL, 0x00f000f000f000f0ULL, 0x00f000f000f000f0ULL, 0x00f000f000f000f0ULL, 0x00f000f000f000f0ULL, 0x00f000f000f000f0ULL, 0x00f000f000f000f0ULL, 0x00f000f000f000f0ULL);
  masks[2] = _mm512_set_epi64(0x0f000f000f000f00ULL, 0x0f000f000f000f00ULL, 0x0f000f000f000f00ULL, 0x0f000f000f000f00ULL, 0x0f000f000f000f00ULL, 0x0f000f000f000f00ULL, 0x0f000f000f000f00ULL, 0x0f000f000f000f00ULL);
  masks[3] = _mm512_set_epi64(0xf000f000f000f000ULL, 0xf000f000f000f000ULL, 0xf000f000f000f000ULL, 0xf000f000f000f000ULL, 0xf000f000f000f000ULL, 0xf000f000f000f000ULL, 0xf000f000f000f000ULL, 0xf000f000f000f000ULL);

  
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
    if (DEBUG || dbg) syslog(LOG_DEBUG,"thread %d: successfully set thread",thread_id);

  // extract from input data structure
  char *in = (char *)d->in;
  char *out = (char *)d->out;
  int nthreads = d->n_threads;  

  /* DO ALL PROCESSING
   
     "in" is input block: NPACKETS * NANTS * (384*2) * 2 pol * r/i. (384*2 is for the two times)
     "out" needs to be in order NPACKETS * (384*2) * 64 * 2 pol * r/i
     parallelize by splitting on NPACKETS axis. 

   */

  // input and output index and extracted data
  int idx = thread_id; // PACKET idx for input and output
  char * proc_data = (char *)malloc(sizeof(char)*(NPACKETS/nthreads)*NANTS*(384*2)*2); // for 4-bit data
  char * fluffed_data = (char *)malloc(sizeof(char)*(NPACKETS/nthreads)*NANTS*(384*2)*2*2); // for 8-bit data
  
  // extract data
  memcpy(proc_data,in+idx*(NPACKETS/nthreads)*NANTS*(384*2)*2,(NPACKETS/nthreads)*NANTS*(384*2)*2);
  if (DEBUG || dbg) syslog(LOG_DEBUG,"thread %d: extracted data",thread_id);
  
  // do fluffing

  /* 
     technique is to use nybble masks to 
     (a) unmask every fourth nybble
     (b) bit shift to left using mm512_slli_epi16
     (c) sign extend by 4 bits using mm512_srai_epi16
     (d) bit shift to right

     Will produce m512 for lower and upper bytes. Then just need to copy into fluffed_data

   */

  // variables
  char * low = (char *)malloc(sizeof(char)*64); // m512
  char * hi = (char *)malloc(sizeof(char)*64); // m512
  __m512i low_m, hi_m;
  unsigned short * low_u = (unsigned short *)(low);
  unsigned short * hi_u = (unsigned short *)(hi);
  __m512i v[4]; // for 4 packed 4-bit numbers

  // input and output
  __m512i proc_m;
  unsigned short * fluffed_u = (unsigned short *)(fluffed_data);

  // numbers to iterate over
  int n_512 = (NPACKETS/nthreads)*NANTS*(384*2)*2/64;

  if (dbg || DEBUG) syslog(LOG_DEBUG,"thread %d: ready to fluff",thread_id);
  
  // let's do it!
  for (int i=0;i<n_512;i++) { // loop over lots of 512 bits

    if (dbg) syslog(LOG_DEBUG,"thread %d: beginning fluff %d",thread_id,i);

    // get input data
    proc_m = _mm512_loadu_si512((proc_data+i*64));
    if (dbg) syslog(LOG_DEBUG,"thread %d: copied data %d",thread_id,i);
    
    // retrieve masks
    for (int j=0;j<4;j++) {
      v[j] = _mm512_and_si512(proc_m, masks[j]);
    }

    if (dbg) syslog(LOG_DEBUG,"thread %d: masked %d",thread_id,i);
    
    // do in place fluffing
    v[0] = _mm512_slli_epi16(v[0], 12);
    v[0] = _mm512_srai_epi16(v[0], 4);
    v[0] = _mm512_srli_epi16(v[0], 8);

    v[1] = _mm512_slli_epi16(v[1], 8);
    v[1] = _mm512_srai_epi16(v[1], 4);

    v[2] = _mm512_slli_epi16(v[2], 4);
    v[2] = _mm512_srai_epi16(v[2], 4);
    v[2] = _mm512_srli_epi16(v[2], 8);

    v[3] = _mm512_srai_epi16(v[3], 4);

    if (dbg) syslog(LOG_DEBUG,"thread %d: in place %d",thread_id,i);

    // make lower and upper 
    low_m = _mm512_or_si512(v[0], v[1]);
    hi_m = _mm512_or_si512(v[2], v[3]);

    if (dbg) syslog(LOG_DEBUG,"thread %d: lower and upper %d",thread_id,i);

    // copy back to bytes
    _mm512_storeu_si512((__m512i *) &low[0], low_m);
    _mm512_storeu_si512((__m512i *) &hi[0], hi_m);

    if (dbg) syslog(LOG_DEBUG,"thread %d: copied lower and upper %d",thread_id,i);
    
    // extract from lower and upper into fluffed
    // there are 32 2-byte unsigned shorts in each of low and hi
    for (int j=0;j<32;j++) {
      fluffed_u[i*64+j*2] = low_u[j];
      fluffed_u[i*64+j*2+1] = hi_u[j];
    }

    if (dbg) syslog(LOG_DEBUG,"thread %d: extracted %d",thread_id,i);
    
  }

  if (dbg || DEBUG) syslog(LOG_DEBUG,"thread %d: fluffed",thread_id);

  memcpy(out + idx*(NPACKETS/nthreads)*(384*2)*NANTS*2*2,fluffed_data,(NPACKETS/nthreads)*(384*2)*NANTS*2*2);
  
  if (dbg || DEBUG) syslog(LOG_DEBUG,"thread %d: done - freeing",thread_id);
  
  // free stuff
  free(proc_data);
  free(fluffed_data);
  free(low);
  free(hi);
  
  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);
  
}


// MAIN

int main (int argc, char *argv[]) {
  
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_reorder_raw", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  // TESTING and initialization
  // threads
  struct data args[16];
  pthread_t threads[16];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t in_key = CAPTURED_BLOCK_KEY;
  key_t out_key = REORDER_BLOCK_KEY;
  
  // command line arguments
  int core = -1;
  int nthreads = 1;
  int bf = 0;
  int arg = 0;
  
  while ((arg=getopt(argc,argv,"c:t:i:o:dqh")) != -1)
    {
      switch (arg)
	{
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
	case 't':
	  if (optarg)
	    {
	      nthreads = atoi(optarg);
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

	case 'q':
	  syslog (LOG_INFO, "Quit here");
	  return EXIT_SUCCESS;
	  
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

  
  // DADA stuff
  
  syslog (LOG_INFO, "creating in and out hdus");
  
  hdu_in  = dada_hdu_create (0);
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to dada buffer in");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to dada buffer in");
    return EXIT_FAILURE;
  }

  hdu_out  = dada_hdu_create (0);
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
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      return EXIT_FAILURE;
    }

  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      return EXIT_FAILURE;
    }


  
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %lu %lu\n",block_size,block_out);
  uint64_t  bytes_read = 0;
  char * block, * output_buffer;
  output_buffer = (char *)malloc(sizeof(char)*block_out);
  memset(output_buffer,0,block_out);
  uint64_t written, block_id;

  // set up

  int observation_complete=0;
  int blocks = 0;
  int started = 0;


  
  syslog(LOG_INFO, "starting observation");

  while (!observation_complete) {

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    if (started==0) {
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }

    // DO STUFF

    // set up data structure
    for (int i=0; i<nthreads; i++) {
      args[i].in = block;
      args[i].out = output_buffer;
      args[i].n_threads = nthreads;
      args[i].thread_id = i;
      args[i].debug = 0;
    }

    if (DEBUG) syslog(LOG_DEBUG,"creating %d threads",nthreads);
    
    for(int i=0; i<nthreads; i++){
      if (pthread_create(&threads[i], &attr, &massage, (void *)(&args[i]))) {
 	syslog(LOG_ERR,"Failed to create massage thread %d\n", i);
      }
    }

    pthread_attr_destroy(&attr);
    if (DEBUG) syslog(LOG_DEBUG,"threads kinda running");
    
    for(int i=0; i<nthreads; i++){
      pthread_join(threads[i], &result);
      if (DEBUG) syslog(LOG_DEBUG,"joined thread %d",i);
    }
    
    // write to output

    written = ipcio_write (hdu_out->data_block, output_buffer, block_out);
    	
    
    if (DEBUG) syslog(LOG_DEBUG, "written block %d",blocks);      
    blocks++;
    

    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  free(output_buffer);

  dsaX_dbgpu_cleanup (hdu_in,0);
  dsaX_dbgpu_cleanup (hdu_out,1);
  
}


