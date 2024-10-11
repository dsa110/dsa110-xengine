// -*- c++ -*-
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
float scfac = 1.0;

// structure for pulse injection
typedef struct {

  int verbose;
  float * block;

} dsaX_pulse_t;

// data to pass to threads
struct tdata {
  unsigned char * data;
  float * pulse;
  int n_threads;
  int thread_id;
};
int cores[4] = {13,14,15,16};

// thread to add pulse to data
void * massage (void *args) {

  struct tdata *d = args;
  int thread_id = d->thread_id;

  // set affinity
  const pthread_t pid = pthread_self();
  const int core_id = cores[thread_id];
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  const int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (set_result != 0)
    syslog(LOG_ERR,"thread %d: setaffinity_np fail",thread_id);
  const int get_affinity = pthread_getaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (get_affinity != 0) 
    syslog(LOG_ERR,"thread %d: getaffinity_np fail",thread_id);
  if (CPU_ISSET(core_id, &cpuset))
    if (DEBUG) syslog(LOG_DEBUG,"thread %d: successfully set thread",thread_id);

  // extract from input
  unsigned char *in = (char *)d->data;
  float * pulse = d->pulse;
  int n_threads = d->n_threads;  

  // do partial addition
  
  float val;
  
  for (int i=(thread_id*NTIMES_P/n_threads);i<((thread_id+1)*NTIMES_P/n_threads);i++) {
    for (int j=0;j<NCHAN_P;j++) {

      val = (float)(in[i*NCHAN_P+j]) + scfac*pulse[i*NCHAN_P+j];
      in[i*NCHAN_P+j] = (unsigned char)(round(val));
      
    }
  }

  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);



}


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
    int tmp_dumpbm = (int)(strtof(strtok(rest, "-"),&endptr));
    if (tmp_dumpbm<0 || tmp_dumpbm>127) tmp_dumpbm=64;
    char * tmp_flnam = strtok(NULL, "-");
    float tmp_snr = (float)(strtof(strtok(NULL, "-"),&endptr));    

    
    if (!dump_pending) {
      strcpy(flnam,tmp_flnam);
      dumpbm = tmp_dumpbm;
      scfac = tmp_snr;
      syslog(LOG_INFO, "control_thread: received command to add pulse %s to beam %d with scfac %g",flnam,dumpbm,scfac);
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
void inject_pulse(unsigned char * data, float * pulse, int bm) {

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
  
  while ((arg=getopt(argc,argv,"c:i:o:k:dh")) != -1)
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

  // set up threads
  struct tdata args[4];
  pthread_t threads[4];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;
  
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
      //inject_pulse(h_data,udpdb.block,dumpbm);
      //fwrite(h_data,1,BUF_SIZE,fout);

      // add pulse with four threads
      for (int i=0;i<4;i++) {
	args[i].data = h_data + dumpbm*NTIMES_P*NCHAN_P;
	args[i].pulse = udpdb.block;
	args[i].n_threads = 4;
	args[i].thread_id = i;
      }

      syslog(LOG_INFO, "creating threads");
      
      for(int i=0; i<4; i++){
	if (pthread_create(&threads[i], &attr, &massage, (void *)(&args[i]))) {
	  syslog(LOG_ERR,"Failed to create massage thread %d\n", i);
	}
      }
      
      pthread_attr_destroy(&attr);
      if (DEBUG) syslog(LOG_DEBUG,"threads kinda running");
      
      for(int i=0; i<4; i++){
	pthread_join(threads[i], &result);
	if (DEBUG) syslog(LOG_DEBUG,"joined thread %d",i);
      }

      
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
  void* fresult=0;
  pthread_join (control_thread_id, &fresult);

  return 0;    
} 
