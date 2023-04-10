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

// print fn
void print_arr(char *ptr, int len) {
  printf("\n[");
  for (int i = 0; i < len; i++) {
    printf(" %08x,", ptr[i]);
  }
  printf(" ]\n");
}

// read and write functions

int write_block(dada_hdu_t* hdu_in) {

  dada_hdu_lock_write(hdu_in);
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  char * data = (char *)malloc(sizeof(char)*block_size);
  memset(data, 0, block_size);
  ipcio_write (hdu_in->data_block, data, block_size);
  free(data);
  dada_hdu_unlock_write (hdu_in);
  
}

int read_block(dada_hdu_t* hdu_in) {

  dada_hdu_lock_read(hdu_in);
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  char * data = (char *)malloc(sizeof(char)*block_size);
  char * block;
  uint64_t  bytes_read, block_id;
  
  block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
  memcpy(data, block, bytes_read);
  print_arr(data, (int)(bytes_read));
  
  free(data);
  ipcio_close_block_read (hdu_in->data_block, bytes_read);
  dada_hdu_unlock_read (hdu_in);
  
}



// MAIN

int main (int argc, char *argv[]) {
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;

  // data block HDU keys
  key_t in_key = TEST_BLOCK_KEY;
  
  // command line arguments
  int arg = 0;
  char *hout;
  hout = (char *)malloc(sizeof(char)*4096);

  
  while ((arg=getopt(argc,argv,"i:h:")) != -1)
    {
      switch (arg)
	{
	case 'i':
	  if (optarg)
	    {
	      sscanf (optarg, "%x", &in_key);
	      break;
	    }
	case 'h':
	  if (optarg)
	    {
	      fileread (optarg, hout, 4096);
	      break;
	    }	 
	}
    }
  
  // DADA stuff  
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  dada_hdu_connect (hdu_in);

  /*
  // deal with header
  dada_hdu_lock_write(hdu_in);
  char * header_out = ipcbuf_get_next_write (hdu_in->header_block);
  memcpy (header_out, hout, 4096);
  ipcbuf_mark_filled (hdu_in->header_block, 4096);
  dada_hdu_unlock_write(hdu_in);
  free(hout);

  dada_hdu_lock_read(hdu_in);
  uint64_t header_size = 0;
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  ipcbuf_mark_cleared (hdu_in->header_block);
  dada_hdu_unlock_read(hdu_in);
  */

  // do four reads and four writes

  while (1) {
  
    printf("writing four blocks... ");
    for (int i=0;i<4;i++) {
      write_block(hdu_in);
      sleep(0.5);
    }
    printf("written\n");
    
    sleep(2);
    
    printf("reading four blocks... ");
    for (int i=0;i<4;i++) {
      read_block(hdu_in);
      sleep(0.5);
    }
    printf("read\n");
    
  }
  
}


