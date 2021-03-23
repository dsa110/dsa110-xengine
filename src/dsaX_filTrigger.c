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

void dsaX_dbgpu_cleanup (dada_hdu_t * in);
int dada_bind_thread_to_core (int core);

FILE *output;

void send_string(char *string) /* includefile */
{
  int len;
  len=strlen(string);
  fwrite(&len, sizeof(int), 1, output);
  fwrite(string, sizeof(char), len, output);
}

void send_float(char *name,float floating_point) /* includefile */
{
  send_string(name);
  fwrite(&floating_point,sizeof(float),1,output);
}

void send_double (char *name, double double_precision) /* includefile */
{
  send_string(name);
  fwrite(&double_precision,sizeof(double),1,output);
}

void send_int(char *name, int integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(int),1,output);
}

void send_char(char *name, char integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(char),1,output);
}


void send_long(char *name, long integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(long),1,output);
}

void send_coords(double raj, double dej, double az, double za) /*includefile*/
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za);
}

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
	   "dsaX_filTrigger [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -i IP to listen to [no default]\n"
	   " -j in_key [default eaea]\n"
	   " -d debug\n"
	   " -n output file name base [no default]\n"
	   " -b beam number of first beam [default 0]\n"
	   " -z respond to zero specnum\n"
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
      specnum = tmps/4;
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
  openlog ("dsaX_filTrigger", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;

  /* port for control commands */
  int control_port = TRIGGER_CONTROL_PORT;

  /* actual struct with info */
  udpdb_t udpdb;
  
  // input data block HDU key
  key_t in_key = 0x0000eaea;

  // command line arguments
  int core = -1;
  int beamn = 0;
  char of[200];
  char foutnam[300];
  char dirnam[300];
  int rz=0;
  int arg=0;

  while ((arg=getopt(argc,argv,"i:c:j:db:n:hz")) != -1)
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
	case 'b':
	  if (optarg)
	    {
	      beamn = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog (LOG_ERR,"ERROR: -b flag requires argument\n");
	      return EXIT_FAILURE;
	    }
	case 'n':
	  if (optarg)
	    {
	      strcpy(of,optarg);
	      break;
	    }
	  else
	    {
	      syslog (LOG_ERR,"ERROR: -n flag requires argument\n");
	      return EXIT_FAILURE;
	    }
	case 'd':
	  DEBUG=1;
	  syslog (LOG_INFO, "Will excrete all debug messages");
	  break;
	case 'z':
	  rz=1;
	  syslog (LOG_INFO, "Will respond to zero trigger");
	  break;
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

  
  // stuff for writing data
  /*
    Data will have [64 beam, time, freq] for each block.
    Need to extract 
   */


  
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  unsigned char * extData = (unsigned char *)malloc(sizeof(unsigned char)*NSAMPS_PER_BLOCK*NCHAN_FIL*NBEAMS_PER_BLOCK);
  uint64_t specs_per_block = NSAMPS_PER_BLOCK;
  uint64_t current_specnum = 0; // updates with each dada block read
  uint64_t start_byte, bytes_to_copy, bytes_copied=0;
  char * in_data;
  uint64_t written=0;
  uint64_t block_id, bytes_read=0;
  int dumping = 0;
  FILE *ofile;
  ofile = fopen("/home/ubuntu/data/dumps.dat","a");
  fprintf(ofile,"starting...\n");
  fclose(ofile);


  // main reading loop
  float pc_full = 0.;
  
  syslog(LOG_INFO, "main: starting observation");

  while (!observation_complete) {
    
    // read a DADA block
    in_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    
    // add delay
    // only proceed if input data block is 80% full
    while (pc_full < 0.8) {
      pc_full = ipcio_percent_full(hdu_in->data_block);
      usleep(100);
    }
    pc_full = 0.;
    
    
    // check for dump_pending
    if (dump_pending) {
      
      // look after hand trigger
      if (specnum==0 && rz==1) {
	
	specnum = current_specnum + 40000;
	
      }
      
      // if this is the first block to dump
      if (specnum > current_specnum && specnum < current_specnum+specs_per_block) {
	
	dumping = 1;
	syslog(LOG_INFO,"dumping is 1 -- first block");
	
	// loop over beams
	bytes_to_copy = (NSAMPS_PER_BLOCK-(specnum-current_specnum))*NCHAN_FIL;
	bytes_copied = bytes_to_copy;
	for (int i=0;i<NBEAMS_PER_BLOCK;i++) {
	  
	  start_byte = i*NSAMPS_PER_BLOCK*NCHAN_FIL + (specnum-current_specnum)*NCHAN_FIL;
	  memcpy(extData + i*NSAMPS_PER_BLOCK*NCHAN_FIL, in_data + start_byte, bytes_to_copy);
	  
	}
	
      }
      
      // if this is the last block to dump from
      if (specnum + NSAMPS_PER_BLOCK > current_specnum && specnum + NSAMPS_PER_BLOCK <= current_specnum + specs_per_block && dumping==1) {	  

	syslog(LOG_INFO,"in second block");
	
	// loop over beams
	bytes_to_copy = NSAMPS_PER_BLOCK*NCHAN_FIL-bytes_copied;
	for (int i=0;i<NBEAMS_PER_BLOCK;i++) {
	  
	  start_byte = i*NSAMPS_PER_BLOCK*NCHAN_FIL;
	  memcpy(extData + i*NSAMPS_PER_BLOCK*NCHAN_FIL + bytes_copied, in_data + start_byte, bytes_to_copy);
	  
	}

	syslog(LOG_INFO,"finished copying");
	
	// DO THE WRITING

	sprintf(dirnam,"mkdir -p %s_%llu",of,specnum*4);
	system(dirnam);
	
	for (int i=0;i<NBEAMS_PER_BLOCK;i++) {
	  
	  sprintf(foutnam,"%s_%llu/%llu_%d.fil",of,specnum*4,specnum*4,beamn+i);
	  output = fopen(foutnam,"wb");
	  
	  send_string("HEADER_START");
	  send_string("source_name");
	  send_string(foutnam);
	  send_int("machine_id",1);
	  send_int("telescope_id",82);
	  send_int("data_type",1); // filterbank data
	  send_double("fch1",1530.0); // THIS IS CHANNEL 0 :)
	  send_double("foff",-0.244140625);
	  send_int("nchans",1024);
	  send_int("nbits",8);
	  send_double("tstart",55000.0);
	  send_double("tsamp",8.192e-6*8.*4.);
	  send_int("nifs",1);
	  send_string("HEADER_END");
	  
	  fwrite(extData + i*NSAMPS_PER_BLOCK*NCHAN_FIL,sizeof(unsigned char),NSAMPS_PER_BLOCK*NCHAN_FIL,output);
	  
	  fclose(output);
	  
	}
	
	syslog(LOG_INFO, "written trigger from specnum %llu TRIGNUM%d DUMPNUM%d %s", specnum, trignum-1, dumpnum, footer_buf);
	ofile = fopen("/home/ubuntu/data/dumps.dat","a");
	fprintf(ofile,"written trigger from specnum %llu TRIGNUM%d DUMPNUM%d %s\n", specnum, trignum-1, dumpnum, footer_buf);
	fclose(ofile);
	
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

  free(extData);
  dsaX_dbgpu_cleanup (hdu_in);

}
