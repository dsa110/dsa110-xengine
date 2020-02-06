// -*- c++ -*-

/* will generate test vector mimicking raw data input */
/* test vector is all real and has ramp in amplitude */
/* different antennas are added by 1, and pols go opposite ways */

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

#include "dada_cuda.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_def.h"

#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

/* global variables */
int quit_threads = 0;
char STATE[20];
int DEBUG = 0;

#define NSUM 25

double gaussrand()
{
  double x = 0;
  int i;
  for(i = 0; i < NSUM; i++)
    x += (double)rand() / RAND_MAX;

  x -= NSUM / 2.0;
  x /= sqrt(NSUM / 12.0);

  return x;
}


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
	   "dsaX_testVec [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -h print usage\n");
}

// MAIN

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_testVec", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  // set CUDA device
  cudaSetDevice(1);
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t out_key = CAPTURE_BLOCK_KEY;
  key_t in_key = TEST_BLOCK_KEY;
  
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
  if (block_out != block_size) {
    syslog(LOG_ERR,"input and output block sizes not the same");
    return EXIT_FAILURE;
  }
  uint64_t  bytes_read = 0;
  char * block;
  uint64_t written, block_id;
  thrust::device_vector<char> d_input(block_size);
  thrust::device_vector<char> d_output(block_out);
  char *dinput = thrust::raw_pointer_cast(d_input.data());
  char *doutput = thrust::raw_pointer_cast(d_output.data());

  // register input hdu with gpu
  //dada_cuda_dbregister(hdu_in);

  // fill test vector
  //  [Time, ANT (NSNAPS), Chan (NCHANG), Ant (3), chan (384), time (2 per packet), pol (2), R/I]
  // all 4 bit
  syslog(LOG_INFO, "generating test vector");
  char * tvec, cvr;
  float tvr;
  uint64_t NVEC;
  NVEC = block_size / (NSNAPS*NCHANG*3*384*2*2);
  float nois[NVEC];
  for (int i=0;i<NVEC;i++) nois[i] = gaussrand();
  tvec = (char *)malloc(sizeof(char)*block_size);
  int antenna, channel, idx=0;
  for (int tt=0;tt<NVEC;tt++) {
    for (int snap=0;snap<NSNAPS;snap++) {
      for (int chang=0;chang<NCHANG;chang++) {
	for (int aant=0;aant<3;aant++) {
	  for (int chan=0;chan<384;chan++) {
	    for (int tim=0;tim<2;tim++) {
	      for (int pol=0;pol<2;pol++) {

		antenna = snap*3+aant;
		channel = chan*384+chan;

		if (pol==0) 
		  tvr = 2*(antenna*1./(3.*NSNAPS)+channel*1./NCHAN)*nois[tt];
		else
		  tvr = 2*(antenna*1./(3.*NSNAPS)+(NCHAN-1.-channel*1.)/NCHAN)*nois[tt];
		
		cvr = ((char)(tvr)) * 16;
		tvec[idx] = (char)(((unsigned char)(cvr)) & ((unsigned char)(240)));		
		
		idx += 1;
		
	      }
	    }
	  }
	}
      }
    }
  }

  
  
  
  // set up

  bool observation_complete=0;
  int blocks = 0;
  int ct = 0;
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

    // write to output
    written = ipcio_write (hdu_out->data_block, tvec, block_out);
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

    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  free(tvec);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


