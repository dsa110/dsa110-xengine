// -*- c++ -*-
/* will run xgpu */
/* assumes input block size is appropriate */
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

#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>

#include "dada_cuda.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_def.h"
#include "cube/cube.h"
#include "xgpu.h"

/* global variables */
int DEBUG = 0;
int cores[8] = {26,27,28,29,20,21,22,23};

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);
int dada_bind_thread_to_core (int core);

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
	 "dsaX_xgpu [options]\n"
	 " -c core   bind process to CPU core [no default]\n"
	 " -d send debug messages to syslog\n"
	 " -t number of threads for reading and writing [default 1]\n"
	 " -i in_key [default REORDER_BLOCK_KEY]\n"
	 " -o out_key [default XGPU_BLOCK_KEY]\n"
	 " -h print usage\n");
}


// data to pass to threads
struct data {
  char * data;
  int n_threads;
  int thread_id;
  int debug;
  ipcio_t * ipc;
  char * input;
  int write;
  uint64_t size;
};

/* thread for data massaging */
void * massage(void *args) {

  // basic stuff
  struct data *d = (data *) args;
  int thread_id = d->thread_id;
  int dbg = d->debug;
  char *data = (char *)d->data;
  int nthreads = d->n_threads;
  uint64_t size = d->size;
  
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

  // do stuff

  uint64_t buf_size = size/nthreads;
  
  // read into data
  if (d->write==0) 
    memcpy(data + thread_id*buf_size, d->input + thread_id*buf_size, buf_size);

  // write to buffer
  if (d->write==1) 
    memcpy(d->ipc->curbuf + thread_id*buf_size, data + thread_id*buf_size, buf_size);
  
  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);
  
  
}

// MAIN

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_xgpu", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t in_key = REORDER_BLOCK_KEY;
  key_t out_key = XGPU_BLOCK_KEY;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  int nthreads = 1;
  
  while ((arg=getopt(argc,argv,"c:t:i:o:dh")) != -1)
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

  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");  
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %d %d\n",block_size,block_out);  
  uint64_t  bytes_read = 0;
  char * block;
  char * output_buffer;
  output_buffer = (char *)malloc(sizeof(char)*block_out);
  uint64_t written, block_id;  

  // threads
  struct data args[16];
  pthread_t threads[16];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;
  
  // set up xgpu

  // register input hdu with gpu
  //dada_cuda_dbregister(hdu_in);

  // structures and definitions
  XGPUInfo xgpu_info;
  int syncOp = SYNCOP_DUMP;
  int xgpu_error = 0;
  xgpuInfo(&xgpu_info);
  XGPUContext context;
  context.array_h = NULL;
  context.matrix_h = NULL;
  xgpu_error = xgpuInit(&context, 0);
  if(xgpu_error) {
    syslog(LOG_ERR, "xGPU error %d", xgpu_error);
    dsaX_dbgpu_cleanup (hdu_in, hdu_out);
    return EXIT_FAILURE;
  }
  ComplexInput *array_h = context.array_h; // this is pinned memory
  Complex *cuda_matrix_h = context.matrix_h;
  memset((char *)array_h,0,2*context.array_len);

  syslog(LOG_INFO,"Set up xgpu with input size %d output size %d",context.array_len,context.matrix_len);
  
  // get things started
  bool observation_complete=0;
  bool started = 0;
  syslog(LOG_INFO, "starting observation");
  int blocks = 0;
  
  while (!observation_complete) {

    if (DEBUG) syslog(LOG_DEBUG,"reading block");    
    
    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
      
    // DO STUFF

    // copy to xgpu input

    // set up data structure
    for (int i=0; i<nthreads; i++) {
      args[i].data = (char *)(array_h);
      args[i].n_threads = nthreads;
      args[i].thread_id = i;
      args[i].debug = 0;
      args[i].input = block;
      args[i].ipc = hdu_in->data_block;
      args[i].write = 0;
      args[i].size = block_size;
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
    
    // run xgpu
    xgpu_error = xgpuCudaXengine(&context, syncOp);
    if(xgpu_error) {
      syslog(LOG_ERR, "xGPU error %d\n", xgpu_error);
      return EXIT_FAILURE;
    }

    if (started==0 && blocks==20) {
      syslog(LOG_INFO,"now in RUN state");
      if (DEBUG) {
	for (int i=100;i<200;i++) {
	  syslog(LOG_DEBUG,"INPUT %hhi %hhi",array_h[i].real,array_h[i].imag);
	  syslog(LOG_DEBUG,"OUTPUT %g %g",(float)(cuda_matrix_h[i].real),(float)(cuda_matrix_h[i].imag));
	}
      }
      started=1;
    }    

    // clear device
    xgpuClearDeviceIntegrationBuffer(&context);
    
    // write to output

    // set up write
    hdu_out->data_block->curbuf = ipcbuf_get_next_write ((ipcbuf_t*)hdu_out->data_block);
    hdu_out->data_block->marked_filled = 0;
    
    // set up data structure
    for (int i=0; i<nthreads; i++) {
      args[i].data = (char *)(cuda_matrix_h);
      args[i].n_threads = nthreads;
      args[i].thread_id = i;
      args[i].debug = 0;
      args[i].ipc = hdu_out->data_block;
      args[i].write = 1;
      args[i].size = block_out;
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

    // finish write
    ipcbuf_mark_filled ((ipcbuf_t*)hdu_out->data_block, block_out);
    //ipcio_check_pending_sod (hdu_out->data_block);
    hdu_out->data_block->marked_filled = 1;

    // finish up
    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);
      
    if (DEBUG) syslog(LOG_DEBUG, "written block %d",blocks);	    
    blocks++;
	
  }

  // finish up
  free(output_buffer);
  //dada_cuda_dbunregister(hdu_in);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


