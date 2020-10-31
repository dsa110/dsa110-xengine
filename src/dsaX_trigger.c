/* Code to read from a single dada buffer, and write to disk upon receiving
a trigger. Uses pthread threads and shared memory to listen. 
Sequence of events:
 - starts null-reading dump buffer, while listening for socket command
   + for N second dump, assume N-second dada blocks
 - receives time-since-start, which is converted into a block_start, byte_start, and block_end and byte_end. Sets dump pending, during which time no commands can be accepted. 
 - Upon seeing dump_pending, read code copies data to output dada buffer, which is plugged into dbdisk. Unsets dump_pending.
*/

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

#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_correlator_udpdb_thread.h"
#include "dsaX_def.h"

/* global variables */
int quit_threads = 0;
int dump_pending = 0;
uint64_t specnum = 0;
uint64_t procnum = 0;
int trignum = 0;
int dumpnum = 0;
char iP[100];
char footer_buf[1024];

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out, multilog_t * log);
int dada_bind_thread_to_core (int core);

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out, multilog_t * log)
{
  
  if (dada_hdu_unlock_read (in) < 0)
    {
      multilog(log, LOG_ERR, "could not unlock read on hdu_in\n");
    }
  dada_hdu_destroy (in);

  if (dada_hdu_unlock_write (out) < 0)
    {
      multilog(log, LOG_ERR, "could not unlock read on hdu_out\n");
    }
  dada_hdu_destroy (out);

  
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_correlator_trigger [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -i IP to listen to [no default]\n"
	   " -n output file name [no default]\n"
	   " -h print usage\n");
}


// Thread to control the dumping of data

void control_thread (void * arg) {

  udpdb_t * ctx = (udpdb_t *) arg;
  multilog(ctx->log, LOG_INFO, "control_thread: starting\n");

  // port on which to listen for control commands
  int port = ctx->control_port;

  // buffer for incoming command strings, and setup of socket
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  char* tbuf = (char *) malloc (sizeof(char) * bufsize);
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
  getaddrinfo(iP,"11223",&hints,&res);
  int fd;
  ssize_t ct;
  char tmpstr;
  char cmpstr = 'p';
  char *endptr;
  uint64_t tmps;
  char * token;
  
  multilog(ctx->log, LOG_INFO, "control_thread: created socket on port %d\n", port);
  
  while (!quit_threads) {
    
    fd = socket(res->ai_family,res->ai_socktype,res->ai_protocol);
    bind(fd,res->ai_addr,res->ai_addrlen);
    memset(buffer,'\0',sizeof(buffer));
    multilog(ctx->log, LOG_INFO, "control_thread: waiting for packet\n");
    ct = recvfrom(fd,buffer,1024,0,(struct sockaddr*)&src_addr,&src_addr_len);
    
    multilog(ctx->log, LOG_INFO, "control_thread: received buffer string %s\n",buffer);
    strcpy(tbuf,buffer);
    trignum++;

    // interpret buffer string
    char * rest = buffer;
    tmps = (uint64_t)(strtoull(strtok_r(rest, "-", &rest),&endptr,0)*16);
    
    if (!dump_pending) {
      //specnum = (uint64_t)(strtoull(buffer,&endptr,0)*16);
      specnum = tmps;
      strcpy(footer_buf,tbuf);
      multilog(ctx->log, LOG_INFO, "control_thread: received command to dump at %llu\n",specnum);
    }
	
    if (dump_pending)
      multilog(ctx->log, LOG_ERR, "control_thread: BACKED UP - CANNOT dump at %llu\n",tmps);
  
    if (!dump_pending) dump_pending = 1;
    
    close(fd);
    
  }

  free (buffer);
  free (tbuf);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: exiting\n");

  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);

}
	    

	
int main (int argc, char *argv[]) {

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  /* port for control commands */
  int control_port = CONTROL_PORT;
  
  /* DADA Logger */
  multilog_t* log = 0;

  /* actual struct with info */
  udpdb_t udpdb;
  
  // input data block HDU key
  key_t in_key = 0x0000eaea;
  key_t out_key = 0x0000fafa;

  // command line arguments
  int core = -1;
  int arg=0;

  // output logging
  FILE *foutp;
  char foutnam[100];
    
  while ((arg=getopt(argc,argv,"i:c:n:h")) != -1)
    {
      switch (arg)
	{
	case 'i':
	  strcpy(iP,optarg);
	  break;
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
	case 'n':
	  sprintf(foutnam,"/home/ubuntu/data/%s_triglog.dat",optarg);
	  foutp=fopen(foutnam,"w");
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // DADA stuff
  
  log = multilog_open ("dsaX_correlator_trigger", 0);
  multilog_add (log, stderr);
  udpdb.log = log;
  udpdb.verbose = 1;
  udpdb.control_port = control_port;
  

  multilog (log, LOG_INFO, "dsaX_correlator_trigger: creating hdus\n");

  // open connection to the in/read DBs
  
  hdu_in  = dada_hdu_create (log);
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    printf ("dsaX_correlator_trigger: could not connect to dada buffer\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    printf ("dsaX_correlator_trigger: could not lock to dada buffer\n");
    return EXIT_FAILURE;
  }

  hdu_out  = dada_hdu_create (log);
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    printf ("dsaX_correlator_trigger: could not connect to output dada buffer\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write(hdu_out) < 0) {
    dsaX_dbgpu_cleanup (hdu_in, hdu_out, log);
    fprintf (stderr, "dsaX_correlator_trigger: could not lock4 to eada buffer\n");
    return EXIT_FAILURE;
  }

  // Bind to cpu core
  if (core >= 0)
    {
      printf("binding to core %d\n", core);
      if (dada_bind_thread_to_core(core) < 0)
	printf("dsaX_correlator_trigger: failed to bind to core %d\n", core);
    }

  int observation_complete=0;

  // stuff for writing data
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  uint64_t specs_per_block = NPACKETS;
  uint64_t specs_per_out = NPACKETS*NOUTBLOCKS;
  uint64_t current_specnum = 0; // updates with each dada block read
  uint64_t start_byte, bytes_to_copy, bytes_copied=0;
  char * out_data = (char *)malloc(sizeof(char)*blocksize);
  char * in_data;
  uint64_t written=0;
  uint64_t block_id, bytes_read=0;
  
  // more DADA stuff - deal with headers
  
  uint64_t header_size = 0;

  // read the header from the input HDU
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      multilog(log ,LOG_ERR, "main: could not read next header\n");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out, log);
      return EXIT_FAILURE;
    }

  // now write the output DADA header
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      multilog(log, LOG_ERR, "could not get next header block [output]\n");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out, log);
      return EXIT_FAILURE;
    }

  // copy the in header to the out header
  memcpy (header_out, header_in, header_size);

  // mark the input header as cleared
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      multilog (log, LOG_ERR, "could not mark header block cleared [input]\n");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out, log);
      return EXIT_FAILURE;
    }

  // mark the output header buffer as filled
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      multilog (log, LOG_ERR, "could not mark header block filled [output]\n");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out, log);
      return EXIT_FAILURE;
    }


  // start control thread
  int rval = 0;
  pthread_t control_thread_id;
  multilog(log, LOG_INFO, "starting control_thread()\n");
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &udpdb);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating control_thread: %s\n", strerror(rval));
    return -1;
  }

  // main reading loop
  float pc_full = 0.;
  
  multilog(log, LOG_INFO, "main: starting observation\n");

  while (!observation_complete) {

       // read a DADA block
      in_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    
      // add delay
      // only proceed if input data block is 80% full
      while (pc_full < 0.85) {
	pc_full = ipcio_percent_full(hdu_in->data_block);
	sleep(1);
      }
      pc_full = 0.;
      
    
      // check for dump_pending
      if (dump_pending) {

	// if this is the first block to dump
	if (specnum > current_specnum && specnum < current_specnum+specs_per_block) {
	  
	  // find start byte and bytes to copy
	  start_byte = 8192*(specnum-current_specnum);
	  bytes_to_copy = blocksize-start_byte;
	  
	  // do copy
	  memcpy(out_data, in_data+start_byte, bytes_to_copy);
	  bytes_copied = bytes_to_copy;
	  
	}

	// if this is the second block in the pair to dump from
	if (specnum < current_specnum && specnum > current_specnum-specs_per_block) {	  

	  // find start byte and bytes to copy
	  start_byte = 0;
	  bytes_to_copy = blocksize-bytes_copied-1024;

	  // do copy
	  memcpy(out_data+bytes_copied, in_data, bytes_to_copy);
	  bytes_copied += bytes_to_copy;

	  // DO THE WRITING
	  written = ipcio_write (hdu_out->data_block, out_data, blocksize);

	  if (written < blocksize)
	    {
	      multilog(log, LOG_INFO, "main: failed to write all data to datablock [output]\n");
	      dsaX_dbgpu_cleanup (hdu_in, hdu_out, log);
	      return EXIT_FAILURE;
	    }
	  multilog(log, LOG_INFO, "main: written trigger from specnum %llu TRIGNUM%d DUMPNUM%d %s\n", specnum, trignum-1, dumpnum, footer_buf);
	  fprintf(foutp,"%d %llu %s\n",dumpnum,specnum,footer_buf);
	  
	  dumpnum++;
	  
	  // reset
	  bytes_copied = 0;
	  dump_pending = 0;
	  
	}

	// if trigger arrived too late
	if (specnum < current_specnum-specs_per_block) {
	  multilog(log, LOG_INFO, "main: trigger arrived too late: specnum %llu, current_specnum %llu\n",specnum,current_specnum);

	  bytes_copied=0;
	  dump_pending=0;

	}

      }

      // update current spec
      current_specnum += specs_per_block;

      // for exiting
      if (bytes_read < blocksize) {
	observation_complete = 1;
	multilog(log, LOG_INFO, "main: finished, with bytes_read %llu < expected %llu\n", bytes_read, blocksize);
      }

      // close block for reading
      ipcio_close_block_read (hdu_in->data_block, bytes_read);

      //    }

  }

  fclose(foutp);

  // close control thread
  multilog(log, LOG_INFO, "joining control_thread\n");
  quit_threads = 1;
  void* result=0;
  pthread_join (control_thread_id, &result);

  free(out_data);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out, log);

}
