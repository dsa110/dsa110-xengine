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
int quit_threads = 0;
char STATE[20];
int DEBUG = 0;

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

/* functor to make map */
struct map_functor
{

  int n;
  map_functor(int _n) : n(_n) {}
  
  __device__
  int operator()(const int x) const {
    
    int i = (int)(x/(NSNAPS*12));
    return x+i*n;
    
  }
}; 

/* KERNEL */

// input is [Time, Ant (NSNAPS snaps), Chan (NCHANG groups), ant (3 per snap), chan (384 per group), time (2 per packet), pol (2), R/I]
// output is [time, frequency, ANT, pol, ri]
// here, ANT=16, frequency=1536
// strictly expect NNATINTS*NCORRINTS time samples per call. Use NNATINTS/2 blocks and NCORRINTS threads
__global__
void massage(char *inpt, char *output) {

  int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index - runs over Time

  int NBYTES_PER_THREAD = NSNAPS*NCHANG*3*384*2*2; // number of bytes per thread
  int inpt_sidx = NBYTES_PER_THREAD*idx; // start idx for input
  int output_sidx = 2*NCHAN*NSNAPS*3*2*2*idx; // start idx for output
  int inpt_idx,output_idx;

  __shared__ char toutp[2*NCHAN*NSNAPS*3*2*2];
  
  for (int i1=0;i1<NSNAPS;i1++) { // Ant
    for (int i2=0;i2<NCHANG;i2++) { // Chan
      for (int i3=0;i3<3;i3++) { // ant
	for (int i4=0;i4<384;i4++) { // chan
	  for (int i5=0;i5<2;i5++) { // time

	    inpt_idx = inpt_sidx + 2 * (i1*NCHANG*3*384*2 + i2*3*384*2 + i3*384*2 + i4*2 + i5);
	    
	    output_idx = i5*NCHAN*NSNAPS*3*2*2 + (i2*384+i4)*NSNAPS*3*2*2 + (i1*3+i3)*2*2;

            // real parts
	    toutp[output_idx] = ((char)(((unsigned char)(inpt[inpt_idx]) & (unsigned char)(15)) << 4))/16;
	    toutp[output_idx+2] = ((char)(((unsigned char)(inpt[inpt_idx+1]) & (unsigned char)(15)) << 4))/16;
	    // imaginary parts
	    toutp[output_idx+1] = ((char)((unsigned char)(inpt[inpt_idx]) & (unsigned char)(240)))/16;
	    toutp[output_idx+3] = ((char)((unsigned char)(inpt[inpt_idx+1]) & (unsigned char)(240)))/16;	    	    

	  }
	}
      }
    }
  }

  __syncthreads();

  for (int i=0;i<2*NCHAN*NSNAPS*3*2*2;i++) {
    output[output_sidx+i] = toutp[i];
  }
}

void usage()
{
fprintf (stdout,
	   "dsaX_xgpu [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
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
  key_t in_key = CAPTURE_BLOCK_KEY;
  key_t out_key = XGPU_BLOCK_KEY;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  
  while ((arg=getopt(argc,argv,"c:dh")) != -1)
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
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // record STATE info
  sprintf(STATE,"NOBUFFER");

  // Bind to cpu core
  if (core >= 0)
    {
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
      syslog(LOG_NOTICE,"bound to core %d", core);
    }  
  
  // for scatter operation
  syslog(LOG_INFO, "preparing map");
  uint64_t block_feed = NREORDERS*NNATINTS*NCORRINTS*NCHAN*NSNAPS*3*2*2;
  thrust::device_vector<int> map(block_feed/NREORDERS);
  thrust::sequence(map.begin(),map.end(),(int)0,(int)1);
  int nnn = NANT*2*2-NSNAPS*3*2*2;
  thrust::transform(map.begin(),map.end(),map.begin(),map_functor(nnn));  
  
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
  sprintf(STATE,"LISTEN");
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");  
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %d %d\n",block_size,block_out);
  // check that these are correct
  // one time sample is NCHAN*NANT*NPOL bytes. Expect NREORDERS*NNATINTS*NCORRINTS times per block.
  if (block_size != NREORDERS*NNATINTS*NCORRINTS*NCHANG*NSNAPS*4608/2) {
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

  // set up reorder
  //char * ob, * ob_all;  
  //ob = (char *)malloc(sizeof(char)*block_feed);
  //ob_all = (char *)malloc(sizeof(char)*NNATINTS*NCORRINTS*NCHAN*32*2*2);
  //memset(ob_all,0,NNATINTS*NCORRINTS*NCHAN*32*2*2);
  thrust::device_vector<char> d_array(NNATINTS*NCORRINTS*NCHAN*NANT*2*2);
  thrust::fill(d_array.begin(),d_array.end(),0);
  thrust::device_vector<char> d_input(block_size);
  thrust::device_vector<char> d_output(block_feed);
  char *dinput = thrust::raw_pointer_cast(d_input.data());
  char *doutput = thrust::raw_pointer_cast(d_output.data());
  
  // get things started
  bool observation_complete=0;
  bool started = 0;
  syslog(LOG_INFO, "starting observation");
  int blocks = 0;

  
  while (!observation_complete) {

    if (DEBUG) syslog(LOG_DEBUG,"reading block");    
    
    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    if (started==0) {
      sprintf(STATE,"RUN");
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }    

    if (DEBUG) syslog(LOG_DEBUG,"reordering");    
    
    // DO STUFF

    // reorder block
    thrust::copy(block,block+block_size,d_input.begin());
    thrust::fill(d_output.begin(),d_output.end(),0);
    massage<<<512, 32>>>(dinput,doutput);
    cudaDeviceSynchronize();
    //thrust::copy(d_output.begin(),d_output.end(),ob);    
    
    // zero matr and mati
    /*for (int i=0;i<XGPU_SIZE;i++) {
      matr[i] = 0.;
      mati[i] = 0.;
      } */   

    if (DEBUG) syslog(LOG_DEBUG,"looping over accums");    
    
    // loop over accumulations
    /*for (int accum=0;accum<NCORRINTS;accum++) {

      // get data
      thrust::copy(ob + accum*NNATINTS*NCHAN*NANT*NPOL, ob + (accum+1)*NNATINTS*NCHAN*NANT*NPOL,(char *)array_h);
    
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
      
    }*/

    // NREORDERS*NNATINTS*NCORRINTS*NCHAN*NSNAPS*3*2*2;

    // loop over reorders, writing each time
    
    for (int reo=0;reo<NREORDERS;reo++) {

      // get data from d_output into array_h via scatter operation
      thrust::scatter(d_output.begin()+reo*NNATINTS*NCORRINTS*NCHAN*NSNAPS*3*2*2,d_output.begin()+(reo+1)*NNATINTS*NCORRINTS*NCHAN*NSNAPS*3*2*2,map.begin(),d_array.begin());     

      if (DEBUG) syslog(LOG_DEBUG,"running xGPU");
      
      // put it in array_h
      thrust::copy(d_array.begin(), d_array.end(), (char *)array_h);

      // run xGPU
      xgpu_error = xgpuCudaXengine(&context, syncOp);                                               
      if(xgpu_error) {
	syslog(LOG_ERR, "xGPU error %d\n", xgpu_error);                                               
	return EXIT_FAILURE;                                                                          
      }
    
      // accumulate
      for (int i=0;i<XGPU_SIZE;i++) {
	matr[i] = cuda_matrix_h[i].real;
	mati[i] = cuda_matrix_h[i].imag;
      }
    
      if (DEBUG) syslog(LOG_DEBUG,"extracting");    
    
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

      if (DEBUG) {
	syslog(LOG_DEBUG, "written block %d",blocks);
	blocks++;
      }

    }
      
    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }


  free(output_buffer);
  free(matr);
  free(mati);
  //free(ob);
  //free(ob_all);
  dada_cuda_dbunregister(hdu_in);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


