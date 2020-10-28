//E_GNU
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

// global variables
int DEBUG = 0;

void dsaX_dbgpu_cleanup (dada_hdu_t * in);


void dsaX_dbgpu_cleanup (dada_hdu_t * in)
{

  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);
  
}



// MAIN

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  multilog_t* log = 0;
  openlog ("dumpfil", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;

  // data block HDU keys
  key_t in_key = 0x0000aaae;
  
  // command line arguments
  int useZ = 1;
  char fnam[100];

  
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

  uint64_t header_size = 0;

  // deal with headers
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "could not read next header");
      dsaX_dbgpu_cleanup (hdu_in);
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in);
      return EXIT_FAILURE;
    }

  
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  syslog(LOG_INFO, "main: have input block size %llu\n",block_size);
  uint64_t  bytes_read = 0;
  uint64_t npackets = 1;
  char * block, * output_buffer;
  uint64_t written, block_id;

  // fill output buffer if file exists
  FILE *fout;
  fout=fopen("test.bin","wb");
  if(fout == NULL)
	{
		printf("Error opening file\n");
		exit(1);
	}

  int observation_complete=0;
  int blocks = 0, started = 0;
  
  syslog(LOG_INFO, "starting observation");


  while (blocks < 10) {

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    if (started==0) {
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }
	fwrite(block, sizeof(char), bytes_read, fout);
    blocks++;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  fclose(fout);
  dsaX_dbgpu_cleanup (hdu_in);
  
}
