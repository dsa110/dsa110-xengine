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
#include "cube/cube.h"
#include "xgpu.h"

/* global variables */
int DEBUG = 0;

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

// kernel for fluffing
// run with 6291456 blocks of 32 threads
__global__ void promoter(char *input, char *output) {

  int idx = blockIdx.x*32 + threadIdx.x;
  char v = input[idx];
  
  output[2*idx] = ((v<<4) & 240) >> 4;
  output[2*idx+1] = v >> 4;

}

void usage()
{
fprintf (stdout,
	 "dsaX_xgpu [options]\n"
	 " -c core   bind process to CPU core [no default]\n"
	 " -d send debug messages to syslog\n"
	 " -i in_key [default REORDER_BLOCK_KEY]\n"
	 " -o out_key [default XGPU_BLOCK_KEY]\n"
	 " -h print usage\n");
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
  
  while ((arg=getopt(argc,argv,"c:i:o:dh")) != -1)
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

  // set up data input for fluffing
  char * h_din = (char *)malloc(sizeof(char)*context.array_len);
  char *d_din, *d_dout;
  cudaMalloc((void **)&d_din, context.array_len*sizeof(char));
  cudaMalloc((void **)&d_dout, 2*context.array_len*sizeof(char)); 

  
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

    // do fluff
    cudaMemcpy(d_din,block,context.array_len*sizeof(char),cudaMemcpyHostToDevice);
    promoter<<<6291456,32>>>(d_din,d_dout);
    //cudaMemcpy((char *)(array_h),d_dout,2*context.array_len*sizeof(char),cudaMemcpyDeviceToHost);        
    cudaDeviceSynchronize();
    
    // run xgpu
    xgpu_error = xgpuCudaXengine(&context, (ComplexInput *)d_dout, syncOp);
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

    written = ipcio_write (hdu_out->data_block, (char *)(cuda_matrix_h), block_out);
    if (written < block_out)
      {
	syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	return EXIT_FAILURE;
      }
    
    // finish up
    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);
      
    if (DEBUG) syslog(LOG_DEBUG, "written block %d",blocks);	    
    blocks++;
	
  }

  // finish up
  free(output_buffer);
  free(h_din);
  cudaFree(d_din);
  cudaFree(d_dout);
  //dada_cuda_dbunregister(hdu_in);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


