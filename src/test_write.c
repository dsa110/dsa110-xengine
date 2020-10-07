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

#define S 4096

// data to pass to threads
struct data {
  char * in;
  int n_threads;
  int thread_id;
  ipcio_t * out;
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
	   " -b connect to bf hdu\n"
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
    if (DEBUG) syslog(LOG_DEBUG,"thread %d: successfully set thread",thread_id);

  // extract from input data structure
  char *in = (char *)d->in;
  //char *out = (char *)d->out;
  int nthreads = d->n_threads;  
  
  // place in out
  int i = thread_id*(S/nthreads);
  //syslog(LOG_INFO,"thread %d: %d",thread_id,i);
  memcpy (d->out->curbuf + i, in + i, S/nthreads);  
  
  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);
  
}


// MAIN

int main (int argc, char *argv[]) {
  
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("test_write", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
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
  dada_hdu_t* hdu_out2 = 0;

  // data block HDU keys
  key_t in_key = CAPTURED_BLOCK_KEY;
  key_t out_key = REORDER_BLOCK_KEY;
  key_t out_key2 = REORDER_BLOCK_KEY2;
  
  // command line arguments
  int core = -1;
  int nthreads = 1;
  int bf = 0;
  int arg = 0;
  
  while ((arg=getopt(argc,argv,"c:t:i:o:dbqh")) != -1)
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
	  
	case 'b':
	  bf=1;
	  syslog (LOG_INFO, "Will write to bf dada hdu");
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
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);
      
      
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
      //dsaX_dbgpu_cleanup (hdu_in, hdu_out, hdu_out2);
      return EXIT_FAILURE;
    }

  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
      //dsaX_dbgpu_cleanup (hdu_in, hdu_out, hdu_out2);
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
      //dsaX_dbgpu_cleanup (hdu_in, hdu_out, hdu_out2);
      return EXIT_FAILURE;
    }

  if (bf) {
    header_out = ipcbuf_get_next_write (hdu_out2->header_block);
    if (!header_out)
      {
	syslog(LOG_ERR, "could not get next header2 block [output]");
	dsaX_dbgpu_cleanup (hdu_in,0);
	dsaX_dbgpu_cleanup (hdu_out,1);
	if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
	//dsaX_dbgpu_cleanup (hdu_in, hdu_out, hdu_out2);
	return EXIT_FAILURE;
      }
    memcpy (header_out, header_in, header_size);
    if (ipcbuf_mark_filled (hdu_out2->header_block, header_size) < 0)
      {
	syslog (LOG_ERR, "could not mark header block2 filled [output]");
	dsaX_dbgpu_cleanup (hdu_in,0);
	dsaX_dbgpu_cleanup (hdu_out,1);
	if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);
	//dsaX_dbgpu_cleanup (hdu_in, hdu_out, hdu_out2);
	return EXIT_FAILURE;
      }
  }

  
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %llu %llu\n",block_size,block_out);
  uint64_t  bytes_read = 0;
  char * block, * output_buffer, * blockie;
  output_buffer = (char *)malloc(sizeof(char)*block_out);
  memset(output_buffer,1,block_out);
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

    // sort out write
    hdu_out->data_block->curbuf = ipcbuf_get_next_write ((ipcbuf_t*)hdu_out->data_block);
    hdu_out->data_block->marked_filled = 0;      
    //blockie = ipcio_open_block_write (hdu_out->data_block, &block_id);
    
    // set up data structure
    for (int i=0; i<nthreads; i++) {
      args[i].in = output_buffer;
      args[i].n_threads = nthreads;
      args[i].thread_id = i;
      args[i].out = hdu_out->data_block;
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

    //written = ipcio_write (hdu_out->data_block, output_buffer, block_out);
    
    // finish write
    ipcbuf_mark_filled ((ipcbuf_t*)hdu_out->data_block, block_out);
    ipcio_check_pending_sod (hdu_out->data_block);
    hdu_out->data_block->marked_filled = 1;      
    //ipcio_close_block_write(hdu_out->data_block, block_out);
    
    if (DEBUG) syslog(LOG_DEBUG, "written block %d",blocks);      
    blocks++;
    

    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  free(output_buffer);

  dsaX_dbgpu_cleanup (hdu_in,0);
  dsaX_dbgpu_cleanup (hdu_out,1);
  if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);	  
  //dsaX_dbgpu_cleanup (hdu_in, hdu_out, hdu_out2);
  
}


