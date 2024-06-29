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
#include "xgpu.h"

// global variables
int DEBUG = 0;
const int n_all = 3194880;
const int nbl = 2080;

// for lookup table generation
// index is position to extract from xgpu array to output (Greg-style) array
void gen_lookup(int * idx_xgpu_in_greg);
void gen_lookup(int * idx_xgpu_in_greg) {

  // get antenna order in xgpu
  int xgpu_ant_1[nbl], xgpu_ant_2[nbl], ct=0;
  for (int i=0;i<64;i++) {
    for (int j=0;j<=i;j++) {
      xgpu_ant_1[ct] = j;
      xgpu_ant_2[ct] = i;
      ct++;
    }
  }

  // get antenna order in Greg
  int gh_ant_1[nbl], gh_ant_2[nbl];
  ct=0;
  for (int i=0;i<64;i++) {
    for (int j=i;j<64;j++) {
      gh_ant_1[ct] = i;
      gh_ant_2[ct] = j;
      ct++;
    }
  }

  // match antenna orders
  for (int i=0;i<nbl;i++) {

    for (int j=0;j<nbl;j++) {
      if (gh_ant_1[i]==xgpu_ant_1[j] && gh_ant_2[i]==xgpu_ant_2[j])
	idx_xgpu_in_greg[i] = j;
    }

  }

}


// for reordering correlations
void reorder_gh(float *input, float *output);
void reorder_gh(float *input, float *output, int * idx_xgpu_in_greg) {

  for (int i=0;i<nbl;i++) {
    for (int j=0;j<384*2*2;j++) {

      output[i*1536+j] = input[idx_xgpu_in_greg[i]*1536+j];

    }
  }
    
}

// for extracting data
// assumes TRIANGULAR_ORDER for mat (f, baseline, pol, ri)
void simple_extract(Complex *mat, float *output);

void simple_extract(Complex *mat, float *output) {

  int in_idx, out_idx;
  for (int bctr=0;bctr<2080;bctr++) {
    for (int pol1=0;pol1<2;pol1++) {

      for (int f=0;f<384;f++) {

	out_idx = 2*((bctr*384+f)*2+pol1);
	in_idx = (2*f*2080+bctr)*4+pol1*3;
	output[out_idx] = 0.5*(mat[in_idx].real + mat[in_idx+8320].real);
	output[out_idx+1] = 0.5*(mat[in_idx].imag + mat[in_idx+8320].imag);

      }
    }
  }

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
	   "dsaX_fake [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default TEST_BLOCK_KEY]\n"
	   " -o out_key [default REORDER_BLOCK_KEY2]\n"
	   " -h print usage\n");
}

// MAIN

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_wrangle", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t in_key = TEST_BLOCK_KEY;
  key_t out_key = REORDER_BLOCK_KEY2;
  
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
  
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %llu %llu\n",block_size,block_out);
  uint64_t  bytes_read = 0;
  char * block;
  uint64_t written, block_id;
  Complex * cblock;
  float *data = (float *)malloc(sizeof(float)*n_all);
  
  
  // set up

  int observation_complete=0;
  int blocks = 0, started = 0;
  
  syslog(LOG_INFO, "starting observation");

  while (!observation_complete) {

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    cblock = (Complex *)(block);
    
    if (started==0) {
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }

    // DO STUFF - from block to summed_vis

    if (DEBUG) syslog(LOG_DEBUG,"extracting...");
    simple_extract((Complex *)(block), data);
    if (DEBUG) syslog(LOG_DEBUG,"extracted!");    

    // write to output
    written = ipcio_write (hdu_out->data_block, (char *)data, block_out);
    if (written < block_out)
      {
	syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	return EXIT_FAILURE;
      }

    if (DEBUG) {
      syslog(LOG_DEBUG, "written block %d",blocks);
      for (int i=0;i<10;i++) {
	syslog(LOG_INFO, "%g", data[i]);
	printf("%g ", data[i]);
	printf("\n");
      }
    }
    blocks++;
    

    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  free(data);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}


