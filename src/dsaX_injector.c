// -*- c++ -*-
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
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"

#include <src/sigproc.h>
#include <src/header.h>


#define NTIMES_P 16384  // # of time samples (assuming 1ms sampling period)
#define NCHAN_P 768	// # of channels on BF node side
#define NBEAMS_P 128	// # of beams on BF side
#define M_P NTIMES_P
#define N_P 32
#define HDR_SIZE 4096
#define BUF_SIZE NTIMES_P*NCHAN_P*NBEAMS_P // size of TCP packet
#define NTHREADS_GPU 32
#define MN 48.0
#define SIG 6.0
#define RMAX 16384
//#define NPERMFLAGS 58
#define NPERMFLAGS 1
#define TBIN 128
#define FBIN 8

// global variables
int DEBUG = 0;
/* global variables */
int quit_threads = 0;
int dump_pending = 0;
int trignum = 0;
char iP[100];
char footer_buf[1024];
char flnam[1024];
int dumpbm;

// structure for pulse injection
typedef struct {

  int verbose;
  float * block;

} dsaX_pulse_t;


// Thread to control the adding of filterbanks
void control_thread (dsaX_pulse_t * ctx) {

  syslog(LOG_INFO, "control_thread: starting");

  // buffer for incoming command strings, and setup of socket
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  char* tbuf = (char *) malloc (sizeof(char) * bufsize);
  memset(buffer, '\0', bufsize);
  const char* whitespace = " ";
  char * command = 0;
  char * args = 0;
  float * tmpblock = (float *)malloc(sizeof(float)*NTIMES_P*NCHAN_P);

  struct addrinfo hints;
  struct addrinfo* res=0;
  memset(&hints,0,sizeof(hints));
  struct sockaddr_storage src_addr;
  socklen_t src_addr_len=sizeof(src_addr);
  hints.ai_family=AF_INET;
  hints.ai_socktype=SOCK_DGRAM;
  getaddrinfo(iP,"11228",&hints,&res);
  int fd;
  ssize_t ct;
  char tmpstr;
  char cmpstr = 'p';
  char *endptr;
  uint64_t tmps;
  char * token;
  double maxval;

  FILE *fin;
  
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
    int tmp_dumpbm = (float)(strtof(strtok(rest, "-"),&endptr));
    if (tmp_dumpbm<0 || tmp_dumpbm>127) tmp_dumpbm=64;
    char * tmp_flnam = strtok(NULL, "-");
    
    if (!dump_pending) {
      strcpy(flnam,tmp_flnam);
      dumpbm = tmp_dumpbm;
      syslog(LOG_INFO, "control_thread: received command to add pulse %s to beam %d",flnam,dumpbm);
      if (!(fin=fopen(flnam,"rb"))) {
	syslog(LOG_INFO,"cannot open %s",flnam);
      }
      else {
	fread(tmpblock,sizeof(float),768*16384,fin);

	// do manipulation of data
	/*maxval = 0.;
	for (int i=0;i<16384*1024;i++) {
	  if (tmpblock[i]>maxval) maxval = tmpblock[i];
	  }*/
	for (int i=0;i<16384;i++) {
	  for (int j=0;j<768;j++) {
	    //ctx->block[i*1024+j] = (float)(tmpblock[j*16384+i]*2.*SIG/maxval);
	    ctx->block[i*768+j] = tmpblock[j*16384+i];
	  }
	}
	
	fclose(fin);
	syslog(LOG_INFO, "control_thread: finished processing pulse - setting dump_pending");
      }
    }
	
    if (dump_pending) {
      syslog(LOG_ERR, "control_thread: BACKED UP - ignoring %s",tbuf);
    }
  
    if (!dump_pending) dump_pending = 1;
    
    close(fd);
    
  }

  free (buffer);
  free (tbuf);
  free(tmpblock);

  if (ctx->verbose)
    syslog(LOG_INFO, "control_thread: exiting");

}

// to actually add pulse to data
void inject_pulse(unsigned char * data, float * pulse, int bm, float scfac) {

  int i0 = bm*NTIMES_P*NCHAN_P;
  float val;
  
  for (int i=0;i<NTIMES_P;i++) {
    for (int j=0;j<NCHAN_P;j++) {

      val = (float)(data[i0+i*NCHAN_P+j]) + scfac*pulse[i*NCHAN_P+j];
      data[i0+i*NCHAN_P+j] = (unsigned char)(round(val));
      
    }
  }
      
}

void usage()
{
  fprintf (stdout,
	   "dsaX_injector [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default dada]\n"
	   " -o out_key [default caca]\n"
	   " -k IP address for injection\n"
	   " -s scale factor for injection [default 1]\n"
	   " -h print usage\n");
}


int main(int argc, char**argv)
{

  // syslog start
  openlog ("dsaX_injector", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  // set cuda device
  cudaSetDevice(1);
  
  // read command line args

  // data block HDU keys
  key_t in_key = 0x0000dada;
  key_t out_key = 0x0000caca;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  float scfac = 1.;
  
  while ((arg=getopt(argc,argv,"c:i:o:s:k:dh")) != -1)
    {
      switch (arg)
	{
	case 'k':
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
	      syslog(LOG_ERR,"-c flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 's':
	  if (optarg)
	    {
	      scfac = atof(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-s flag requires argument");
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

  dsaX_pulse_t udpdb;
  udpdb.verbose = DEBUG;
  float * pulsedata = (float *)malloc(sizeof(float)*16384*768);
  udpdb.block = pulsedata;
  
  // CONNECT AND READ FROM BUFFER

  dada_hdu_t* hdu_in = 0;	// header and data unit
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to input buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to input buffer");
    return EXIT_FAILURE;
  }

  if (DEBUG) syslog(LOG_INFO,"connected to input buffer");
  
  uint64_t header_size = 0;
  // read the header from the input HDU
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  
  // mark the input header as cleared
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0){
    syslog (LOG_ERR,"could not mark header as cleared");
    return EXIT_FAILURE;
  }
  
  uint64_t block_id, bytes_read = 0;
  unsigned char *cin_data;
	     	
  // OUTPUT BUFFER
  dada_hdu_t* hdu_out = 0;
  hdu_out  = dada_hdu_create ();
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog (LOG_ERR,"flagged_data: could not connect to dada buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write (hdu_out) < 0) {
    syslog (LOG_ERR,"flagged_data: could not lock to dada buffer");
    return EXIT_FAILURE;
  }

  if (DEBUG) syslog(LOG_INFO,"connected to output");
  
  
  //// OUTPUT BUFFER
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  header_size = HDR_SIZE;
  if (!header_out)
    {
      syslog(LOG_ERR,"couldn't read header_out");
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      return EXIT_FAILURE;
    }
  uint64_t written=0;

  if (DEBUG) syslog(LOG_INFO,"copied header");
  
  ////////////////		

  // declare stuff for host
  unsigned char * h_data = (unsigned char *)malloc(sizeof(unsigned char)*NBEAMS_P*NTIMES_P*NCHAN_P);
  int blockn=0;
  
  // start control thread                                                                                                                                                     
  int rval = 0;
  pthread_t control_thread_id;
  syslog(LOG_INFO, "starting control_thread()");
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &udpdb);
  if (rval != 0) {
    syslog(LOG_ERR, "Error creating control_thread: %s", strerror(rval));
    return -1;
  }
  
  //FILE *fout;
  //fout=fopen("tmp.tmp","wb");
  
  // put rest of the code inside while loop
  while (1) {	
    
    // read a DADA block
    cin_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    memcpy(h_data,cin_data,NBEAMS_P*NTIMES_P*NCHAN_P*sizeof(unsigned char));
    
    blockn++;
    
    if (DEBUG) syslog(LOG_INFO,"read block");

    // check whether we want to add pulse
    if (dump_pending) {

      syslog(LOG_INFO, "adding pulse %s to beam %d", flnam, dumpbm);

      // add pulse
      inject_pulse(h_data,udpdb.block,dumpbm,scfac);
      //fwrite(h_data,1,BUF_SIZE,fout);
 
      syslog(LOG_INFO, "added %s to beam %d", flnam, dumpbm);
	  
      dump_pending=0;
      
    }
    
    // write to buffer
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    if (DEBUG) syslog(LOG_DEBUG,"closed read block");		    
    written = ipcio_write (hdu_out->data_block, (char *)(h_data), BUF_SIZE);
    if (written < BUF_SIZE)
      {
	syslog(LOG_ERR,"write error");
	return EXIT_FAILURE;
      }
  

    if (DEBUG) syslog(LOG_INFO,"done with round");
    

  }

  // close control thread
  syslog(LOG_INFO, "joining control_thread");
  // close threads
  syslog(LOG_INFO, "joining control_thread");
  quit_threads = 1;
  void* result=0;
  pthread_join (control_thread_id, &result);

  return 0;    
} 
