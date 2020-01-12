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
int quit_threads = 0;
char STATE[20];
int DEBUG = 0;
char iP[100];

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);
int dada_bind_thread_to_core (int core);
void simple_extract (float *matr, float *mati, float *output);

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
	   " -i IP to listen on for control commands [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -h print usage\n");
}

// assumes TRIANGULAR_ORDER for mat (f, baseline, pol, ri)
// based on xGPU xgpuExtractMatrix in cpu_util.c
// output order is (baseline, frequency, pol, r/i)
void simple_extract(float *matr, float *mati, float *output) {

  int in_idx, out_idx;
  for (int f=0;f<NCHAN;f++) {
    for (int bctr=0;bctr<NBASE;bctr++) {
      for (int pol1=0;pol1<NPOL;pol1++) {
      
	in_idx = (f*NBASE+bctr)*4+pol1*3;
	out_idx = 2*((bctr*NCHAN+f)*2+pol1);
	
	output[out_idx] = matr[in_idx];
	output[out_idx+1] = mati[in_idx];
	
      }
    }
  }

}


/* THREADS */

// CONTROL THREAD

void control_thread () {

  syslog(LOG_INFO, "control_thread: starting");

  // port on which to listen for control commands
  int port = XGPU_CONTROL_PORT;
  char sport[10];
  sprintf(sport,"%d",port);

  // buffer for incoming command strings, and setup of socket
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  memset(buffer, '\0', bufsize);
  const char* whitespace = " ";
  char * command = 0;
  char * args = 0;

  struct addrinfo hints;
  struct addrinfo* res=0;
  memset(&hints,0,sizeof(hints));
  struct sockaddr_storage src_addr;
  socklen_t src_addr_len=sizeof(src_addr);
  hints.ai_family=AF_INET;
  hints.ai_socktype=SOCK_DGRAM;
  getaddrinfo(iP,sport,&hints,&res);
  int fd;
  ssize_t ct;
  char tmpstr;
  char cmpstr = 'p';
  char *endptr;
  uint64_t tmps;
  char * token;
  
  syslog(LOG_INFO, "control_thread: created socket on port %d", port);
  
  while (!quit_threads) {
    
    fd = socket(res->ai_family,res->ai_socktype,res->ai_protocol);
    bind(fd,res->ai_addr,res->ai_addrlen);
    memset(buffer,'\0',sizeof(buffer));
    syslog(LOG_INFO, "control_thread: waiting for packet");
    ct = recvfrom(fd,buffer,1024,0,(struct sockaddr*)&src_addr,&src_addr_len);
    
    syslog(LOG_INFO, "control_thread: received buffer string %s",buffer);

    // INTERPRET BUFFER STRING
    // NOTHING TO RECEIVE AT THE MOMENT
    // TODO
    
    
    close(fd);
    
  }

  free (buffer);

  syslog(LOG_INFO, "control_thread: exiting");

  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);

}


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
  
  while ((arg=getopt(argc,argv,"c:d:i:h")) != -1)
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
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	case 'i':
	  if (optarg)
	    {	      
	      strcpy(iP,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-i flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // record STATE info
  sprintf(STATE,"NOBUFFER");

  // START THREADS
  
  // start control thread
  int rval = 0;
  pthread_t control_thread_id, stats_thread_id;
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread);
  if (rval != 0) {
    syslog(LOG_ERR, "Error creating control_thread: %s", strerror(rval));
    return -1;
  }
  syslog(LOG_NOTICE, "Created control thread, listening on %s:%d",iP,REORDER_CONTROL_PORT);

  // Bind to cpu core
  if (core >= 0)
    {
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
      syslog(LOG_NOTICE,"bound to core %d", core);
    }

  
  // DADA stuff
  
  syslog (LOG_INFO, "creating in and out hdus");

  hdu_in  = dada_hdu_create (log);
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to dada buffer in");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to dada buffer in");
    return EXIT_FAILURE;
  }

  hdu_out  = dada_hdu_create (log);
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
  sprintf(STATE,"LISTEN");
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");  
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %d %d\n",block_size,block_out);
  // check that these are correct
  // one time sample is NCHAN*NANT*NPOL bytes. Expect NNATINTS*NCORRINTS times per block.
  if (block_size != NNATINTS*NCORRINTS*NCHAN*NANT*NPOL) {
    syslog(LOG_ERR,"wrong block_size %"PRIu64" in input",block_size);
    return EXIT_FAILURE;
  }
  // output is NBASE*NCHAN*NPOL*8 bytes (baseline, frequency, pol, r/i)
  if (block_out != NBASE*NCHAN*NPOL*8) {
    syslog(LOG_ERR,"wrong block_out %"PRIu64" in output",block_out);
    return EXIT_FAILURE;
  }
  

  uint64_t  bytes_read = 0;
  char * block;
  float * output_buffer;
  output_buffer = (float *)malloc(sizeof(float)*block_out/4);
  uint64_t written, block_id;  
  float * matr, * mati;
  matr = (float *)malloc(sizeof(float)*XGPU_SIZE);
  mati = (float *)malloc(sizeof(float)*XGPU_SIZE);

  // register input hdu with gpu
  dada_cuda_dbregister(hdu_in);

  // set up XGPU
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
  float *output_vis = (float *)malloc(sizeof(float)*XGPU_SIZE);
  memset((char *)array_h,0,2*context.array_len);

  // get things started
  bool observation_complete=0;
  bool started = 0;
  syslog(LOG_INFO, "starting observation");

  while (!observation_complete) {

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    if (started==0) {
      sprintf(STATE,"RUN");
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }
    
    // DO STUFF

    // zero matr and mati
    for (int i=0;i<XGPU_SIZE;i++) {
      matr[i] = 0.;
      mati[i] = 0.;
    }
    
    // loop over accumulations
    for (int accum=0;accum<NCORRINTS;accum++) {

      // get data
      thrust::copy(block + accum*NNATINTS*NCHAN*NANT*NPOL, block + (accum+1)*NNATINTS*NCHAN*NANT*NPOL,(char *)array_h);
    
      // run xGPU
      xgpu_error = xgpuCudaXengine(&context, syncOp);
      if(xgpu_error) {
	syslog(LOG_ERR, "xGPU error %d\n", xgpu_error);
	return EXIT_FAILURE;
      }

      // accumulate
      for (int i=0;i<XGPU_SIZE;i++) {
	matr[i] += cuda_matrix_h[i].real;
	mati[i] += cuda_matrix_h[i].imag;
      }
	
    }

    // simple extract
    simple_extract(matr,mati,output_buffer);
    
    // write to output
    written = ipcio_write (hdu_out->data_block, (char *)output_buffer, block_out);
    if (written < block_out)
      {
	syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	return EXIT_FAILURE;
      }

    if (DEBUG) 
      syslog(LOG_DEBUG, "written block");

    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  // close threads
  syslog(LOG_INFO, "joining control_thread and stats_thread");
  quit_threads = 1;
  void* result=0;
  pthread_join (control_thread_id, &result);

  free(output_buffer);
  free(matr);
  free(mati);
  dada_cuda_dbunregister(hdu_in);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


