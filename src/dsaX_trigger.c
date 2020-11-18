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
#include <syslog.h>

#include "dsaX_capture.h"
#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
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
      syslog(LOG_ERR, "could not unlock read on hdu_out");
    }
  dada_hdu_destroy (out);

  
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_correlator_trigger [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -i IP to listen to [no default]\n"
	   " -j in_key [default eaea]\n"
	   " -o out_key [default fafa]\n"
	   " -d debug\n"
	   " -n output file name [no default]\n"
	   " -h print usage\n");
}


// Thread to control the dumping of data

void control_thread (void * arg) {

  udpdb_t * ctx = (udpdb_t *) arg;
  syslog(LOG_INFO, "control_thread: starting");

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
  getaddrinfo(iP,"11227",&hints,&res);
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
    strcpy(tbuf,buffer);
    trignum++;

    // interpret buffer string
    char * rest = buffer;
    tmps = (uint64_t)(strtoull(strtok_r(rest, "-", &rest),&endptr,0));
    
    if (!dump_pending) {
      //specnum = (uint64_t)(strtoull(buffer,&endptr,0)*16);
      specnum = tmps;
      strcpy(footer_buf,tbuf);
      syslog(LOG_INFO, "control_thread: received command to dump at %llu",specnum);
    }
	
    if (dump_pending)
      syslog(LOG_ERR, "control_thread: BACKED UP - CANNOT dump at %llu",tmps);
  
    if (!dump_pending) dump_pending = 1;
    
    close(fd);
    
  }

  free (buffer);
  free (tbuf);

  if (ctx->verbose)
    syslog(LOG_INFO, "control_thread: exiting");

  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);

}
	    

	
int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_trigger", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  /* port for control commands */
  int control_port = TRIGGER_CONTROL_PORT;

  /* actual struct with info */
  udpdb_t udpdb;
  
  // input data block HDU key
  key_t in_key = 0x0000eaea;
  key_t out_key = 0x0000fafa;

  // command line arguments
  int core = -1;
  int arg=0;

  while ((arg=getopt(argc,argv,"i:c:j:o:d:h")) != -1)
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
	      syslog (LOG_ERR,"ERROR: -c flag requires argument\n");
	      return EXIT_FAILURE;
	    }
	case 'd':
	  DEBUG=1;
	  syslog (LOG_INFO, "Will excrete all debug messages");
	  break;
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
	case 'j':
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
	      syslog(LOG_ERR,"-j flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // DADA stuff
  
  udpdb.verbose = DEBUG;
  udpdb.control_port = control_port;
  
    // start control thread
  int rval = 0;
  pthread_t control_thread_id;
  syslog(LOG_INFO, "starting control_thread()");
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &udpdb);
  if (rval != 0) {
    syslog(LOG_ERR, "Error creating control_thread: %s", strerror(rval));
    return -1;
  }

  
  syslog (LOG_INFO, "creating hdus");

  // open connection to the in/read DBs
  
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to dada buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to dada buffer");
    return EXIT_FAILURE;
  }

  hdu_out  = dada_hdu_create ();
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog (LOG_ERR,"could not connect to output dada buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write(hdu_out) < 0) {
    dsaX_dbgpu_cleanup (hdu_in, hdu_out);
    syslog (LOG_ERR,"could not lock4 to eada buffer");
    return EXIT_FAILURE;
  }

  // Bind to cpu core
  if (core >= 0)
    {
      syslog(LOG_INFO,"binding to core %d", core);
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
    }

  int observation_complete=0;
  
  // more DADA stuff - deal with headers
  
  uint64_t header_size = 0;

  // read the header from the input HDU
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "main: could not read next header");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  // now write the output DADA header
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  // copy the in header to the out header
  memcpy (header_out, header_in, header_size);

  // mark the input header as cleared
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared [input]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  // mark the output header buffer as filled
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  // stuff for writing data
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  uint64_t specs_per_block = 2048;
  uint64_t specs_per_out = 2048*NOUTBLOCKS;
  uint64_t current_specnum = 0; // updates with each dada block read
  uint64_t start_byte, bytes_to_copy, bytes_copied=0;
  char * out_data = (char *)malloc(sizeof(char)*block_out);
  char * in_data;
  uint64_t written=0;
  uint64_t block_id, bytes_read=0;
  int dumping = 0;
  


  // main reading loop
  float pc_full = 0.;
  
  syslog(LOG_INFO, "main: starting observation");

  while (!observation_complete) {

       // read a DADA block
      in_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    
      // add delay
      // only proceed if input data block is 80% full
      while (pc_full < 0.80) {
	pc_full = ipcio_percent_full(hdu_in->data_block);
	usleep(100);
      }
      pc_full = 0.;
      
    
      // check for dump_pending
      if (dump_pending) {

	// if this is the first block to dump
	if (specnum > current_specnum && specnum < current_specnum+specs_per_block) {

	  dumping = 1;
	  
	  // find start byte and bytes to copy
	  start_byte = 4608*NSNAPS*(specnum-current_specnum);
	  bytes_to_copy = block_size-start_byte;
	  
	  // do copy
	  memcpy(out_data, in_data+start_byte, bytes_to_copy);
	  bytes_copied = bytes_to_copy;
	  
	}

	// if this is one of the middle blocks to dump from
	if (specnum < current_specnum && specnum + specs_per_out > current_specnum + specs_per_block && dumping==1) {

	  // do copy
	  memcpy(out_data + bytes_copied, in_data, block_size);
	  bytes_copied += block_size;

	}

	// if this is the last block to dump from
	if (specnum + specs_per_out > current_specnum && specnum + specs_per_out <= current_specnum + specs_per_block && dumping==1) {	  

	  // find start byte and bytes to copy
	  bytes_to_copy = block_out-bytes_copied;

	  // do copy
	  memcpy(out_data+bytes_copied, in_data, bytes_to_copy);

	  // DO THE WRITING
	  written = ipcio_write (hdu_out->data_block, out_data, block_out);

	  if (written < block_out)
	    {
	      syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	      return EXIT_FAILURE;
	    }
	  syslog(LOG_INFO, "written trigger from specnum %llu TRIGNUM%d DUMPNUM%d %s\n", specnum, trignum-1, dumpnum, footer_buf);
	  
	  dumpnum++;
	  
	  // reset
	  bytes_copied = 0;
	  dump_pending = 0;
	  dumping=0;
	  
	}

	// if trigger arrived too late
	if (specnum < current_specnum-specs_per_block && dumping==0 && dump_pending==1) {
	  syslog(LOG_INFO, "trigger arrived too late: specnum %llu, current_specnum %llu",specnum,current_specnum);

	  bytes_copied=0;
	  dump_pending=0;

	}

	
      }

      // update current spec
      if (DEBUG) syslog(LOG_INFO,"current_specnum %llu",current_specnum);
      current_specnum += specs_per_block;
      

      // for exiting
      if (bytes_read < block_size) {
	observation_complete = 1;
	syslog(LOG_INFO, "main: finished, with bytes_read %llu < expected %llu\n", bytes_read, block_size);
      }

      // close block for reading
      ipcio_close_block_read (hdu_in->data_block, bytes_read);


  }


  // close control thread
  syslog(LOG_INFO, "joining control_thread");
  quit_threads = 1;
  void* result=0;
  pthread_join (control_thread_id, &result);

  free(out_data);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);

}
