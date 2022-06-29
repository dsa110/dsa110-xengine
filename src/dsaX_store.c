/* Code to read from a raw data buffer and write to disk */

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
#include <string.h>
#include <unistd.h>
#include <netdb.h>
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

void dsaX_dbgpu_cleanup (dada_hdu_t * in);
int dada_bind_thread_to_core (int core);

void dsaX_dbgpu_cleanup (dada_hdu_t * in)
{
  
  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_dbdisk [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -k in_key [default fafa]\n"
	   " -h print usage\n");
}


int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_store", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  
  // input data block HDU key
  key_t in_key = 0x0000fafa;

  // command line arguments
  uint64_t blocksize;
  uint64_t bout = 32*NSNAPS*4608; // output block size - assume input is a multiple.
  int core = -1;
  int arg=0;

  while ((arg=getopt(argc,argv,"c:k:h")) != -1)
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
	      printf ("ERROR: -c flag requires argument\n");
	      return EXIT_FAILURE;
	    }
	case 'k':
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
	      syslog(LOG_ERR,"-k flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // DADA stuff

  // open connection to the in/read DB
  
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to input buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"dsaX_correlator_copy: could not lock to input buffer");
    return EXIT_FAILURE;
  }
  
  // Bind to cpu core
  if (core >= 0)
    {
      syslog(LOG_INFO,"binding to core %d", core);
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"dsaX_correlator_copy: failed to bind to core %d",core);
    }
  
  // more DADA stuff - deal with headers
  
  uint64_t header_size = 0;

  // read the header from the input HDU
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "main: could not read next header");
      dsaX_dbgpu_cleanup (hdu_in);
      return EXIT_FAILURE;
    }
  
  // mark the input header as cleared
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared [input]");
      dsaX_dbgpu_cleanup (hdu_in);
      return EXIT_FAILURE;
    }

  int observation_complete=0;

  // stuff for writing data
  blocksize = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  char * cpbuf = (char *)malloc(sizeof(char)*blocksize);
  char * outbuf = (char *)malloc(sizeof(char)*bout);
  int ngulps = (int)(blocksize/bout);
  int gulp = 0, wseq = 0;;
  char *in_data;
  uint64_t written=0, written2=0;
  uint64_t block_id, bytes_read=0;
  FILE *fout;
  char fnam[100];
  

  syslog(LOG_INFO, "have ngulps %d, blocksize %llu, bout %llu",ngulps,blocksize,bout);

  
  // main reading loop

  syslog(LOG_INFO, "main: starting read");

  while (!observation_complete) {

    // read a DADA block
    in_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    // copy
    memcpy(cpbuf, in_data, blocksize);
    syslog(LOG_INFO, "starting new write (seq %d)",wseq);

    // open file for writing
    sprintf(fnam,"/home/ubuntu/data/fl_%d.out",wseq);
    fout = fopen(fnam,"wb");
    for (gulp=0;gulp<ngulps;gulp++) {

      // copy to outbuf
      memcpy(outbuf, cpbuf+gulp*bout, bout);

      // write
      usleep(40000);
      fwrite(outbuf, 1, bout, fout);

    }
    fclose(fout);
    wseq++;
    syslog(LOG_INFO, "main: finished new write to file %s",fnam);
    
    // for exiting
    if (bytes_read < blocksize) {
      observation_complete = 1;
      syslog(LOG_INFO, "main: finished, with bytes_read %llu < expected %llu", bytes_read, blocksize);
    }

    // close block for reading
    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }
  
  free(cpbuf);
  free(outbuf);
  dsaX_dbgpu_cleanup (hdu_in);
  
}
  
