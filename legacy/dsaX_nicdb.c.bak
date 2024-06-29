/*
https://dzone.com/articles/parallel-tcpip-socket-server-with-multi-threading

gcc -o test_ipcbuf test_ipcbuf.c -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

the plan is to have NCLIENTS threads listening on different threads. 
each time data comes over the first 8 bytes consist of the channel group and time sequence as two ints
the rest is a NSAMPS_PER_BLOCK*NBEAMS_PER_TRANSMIT*NW char array that needs to be arranged correctly
The output must be [NBEAMS_PER_BLOCK, NSAMPS_PER_BLOCK, NCHAN_FIL]. 

After a block is full, the data need to be written out (data rate 525 Mb/s)
The number of receives before switching blocks is NCLIENTS*NSAMPS_PER_BLOCK/NSAMPS_PER_TRANSMIT. 
switch block when one block is being written out

*/
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

// global variables
int DEBUG = 0;
int blockct = 0; // to count how many writes to block. max is NSAMPS_PER_BLOCK*NBEAMS_PER_BLOCK*NW
int block_switch = 0; // 0 means write to output1, write out output2.
int cores[16] = {3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28}; // to bind threads to
char iP[100];

// structure to pass to threads
struct data
{
  char * output1;
  char * output2;
  uint16_t tport;
  int thread_id;
};

// function prototypes
void dsaX_dbgpu_cleanup (dada_hdu_t * out);
int dada_bind_thread_to_core (int core);

void dsaX_dbgpu_cleanup (dada_hdu_t * out)
{

  if (dada_hdu_unlock_write (out) < 0)
    {
      syslog(LOG_ERR, "could not unlock write on hdu_out");
    }
  dada_hdu_destroy (out);
  
}


// receive process - runs infinite loop
void * process(void * ptr)
{

  // arguments from structure
  struct data *d = ptr;
  int thread_id = d->thread_id;
  char *output1 = (char *)d->output1;
  char *output2 = (char *)d->output2;
  uint16_t tport = d->tport;
  
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
    if (DEBUG) syslog(LOG_INFO,"thread %d: successfully set thread",thread_id);

  // set up socket
  int sock = -1, conn = -1;
  struct sockaddr_in address, cli;

  /* create socket */
  sock = socket(AF_INET, SOCK_STREAM, 0);
  if (DEBUG) syslog(LOG_INFO,"thread %d: opened socket",thread_id);
  memset(&address, 0, sizeof(struct sockaddr_in));
  address.sin_family = AF_INET;
  inet_pton(AF_INET, iP, &(address.sin_addr));
  //address.sin_addr.s_addr = inet_addr("127.0.0.1");
  address.sin_port = htons(tport);
  if (DEBUG) syslog(LOG_INFO,"thread %d: socket ready",thread_id);
  if (bind(sock, (struct sockaddr *)&address, sizeof(struct sockaddr_in)) < 0) {
    syslog(LOG_ERR,"thread %d: cannot bind to port",thread_id);
    exit(1);
  }
  if (DEBUG) syslog(LOG_INFO,"thread %d: socket bound",thread_id);
  listen(sock, 5);
  if (DEBUG) syslog(LOG_INFO,"thread %d: socket listening on port %d",thread_id,tport);
  
  // accept connection
  socklen_t cli_len=sizeof(struct sockaddr);
  conn = accept(sock, (struct sockaddr *) &cli, &cli_len);
  if (conn<0) {
    syslog(LOG_ERR,"thread %d: error accepting connection",thread_id);
    exit(1);
  }
  syslog(LOG_INFO,"thread %d: accepted connection",thread_id);

  // data buffer and other variables
  char * buffer = (char *)malloc((8+NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW)*sizeof(char));
  char * dblock = (char *)malloc((8+NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW)*sizeof(char));
  int *ibuf, chgroup, tseq, oidx, iidx;
  int remain_data, outptr, len;
  int i0;
  
  // infinite loop 
  while (1) {
  
    /* read message */
    // read to buffer until all is read
    remain_data =(int)(8+NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW);
    outptr=0;

    /*
    while (((len = recv(conn, dblock, remain_data, 0)) > 0) && (remain_data > 0)) {
    memcpy(buffer+outptr, dblock, len);
      remain_data -= len;
      outptr += len;
      //syslog(LOG_INFO,"Received %d of %d bytes",outptr,8+NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW);
      }*/
    //recvlen = read(sock, buffer, sizeof(buffer));
    ibuf = (int *)(buffer);
    len = recv(conn, dblock, remain_data, MSG_WAITALL);
    memcpy(buffer, dblock, len);
    remain_data -= len;
    if (remain_data != 0)
      syslog(LOG_ERR,"thread %d: only received %d of %d",thread_id,len,(int)(8+NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW));
    
    if (remain_data==0) {
    
      // get channel group and time sequence
      chgroup = ibuf[0]; // from 0-15
      tseq = ibuf[1]; // continuous iterate over transmits
      if (DEBUG) syslog(LOG_INFO,"thread %d: read message with chgroup %d tseq %d blockct %d",thread_id,chgroup,tseq,blockct);
      tseq = (tseq * 128) % 4096; // place within output
      
      // output order is [beam, time, freq]. input order is [beam, time, freq], but only a subset of freqs
      i0 = 8;
      for (int i=0;i<NBEAMS_PER_BLOCK;i++) {
	for (int j=0;j<NSAMPS_PER_TRANSMIT;j++) {	
	  for (int k=0;k<NW;k++) {
	    
	    oidx = i*NSAMPS_PER_BLOCK*NCHAN_FIL + (tseq+j)*NCHAN_FIL + CHOFF/8 + chgroup*NW + k;
	    //iidx = 8 + i0;
	    
	    if (block_switch==0) output1[oidx] = buffer[i0];
	    if (block_switch==1) output2[oidx] = buffer[i0];

	    i0++;
	    
	  }
	}
      }
      
      // iterate blockct
      blockct++;

    }

  }

  /* close socket and clean up */
  close(sock);
  free(buffer);
  free(dblock);
  pthread_exit(0);
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_nicdb [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -f header file [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -o out_key [default BEAMCAPTURE_BLOCK_KEY]\n"
	   " -i IP address\n"
	   " -h print usage\n");
}


// main part of program 
int main(int argc, char ** argv)
{
    
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_nicdb", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  // threads
  struct data args[16];
  pthread_t threads[16];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t out_key = BEAMCAPTURE_BLOCK_KEY;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  char fnam[200];
  
  while ((arg=getopt(argc,argv,"c:f:o:i:dh")) != -1)
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
	case 'f':
	  if (optarg)
	    {
	      strcpy(fnam,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'd':
	  DEBUG=1;
	  syslog (LOG_INFO, "Will excrete all debug messages");
	  break;
	case 'i':
	  strcpy(iP,optarg);
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

  // deal with headers
  uint64_t header_size = 4096;
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  FILE *fin;
  if (!(fin=fopen(fnam,"rb"))) {
    syslog(LOG_ERR,"cannot open dada header file %s",fnam);
    return EXIT_FAILURE;
  }
  fread(header_out, 4096, 1, fin);
  fclose(fin);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_out);
      return EXIT_FAILURE;
    }  
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_out);
      return EXIT_FAILURE;
    }
    
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have output block sizes %llu\n",block_out);
  uint64_t  bytes_read = 0;
  char *output1, *output2;
  output1 = (char *)malloc(sizeof(char)*block_out);
  output2 = (char *)malloc(sizeof(char)*block_out);
  memset(output1,0,block_out);
  memset(output2,0,block_out);
  uint64_t written, block_id;

  // set up threads
  
  // set up data structure
  for (int i=0; i<NCLIENTS; i++) {
    args[i].output1 = output1;
    args[i].output2 = output2;
    args[i].thread_id = i;
    args[i].tport = FIL_PORT0 + (uint16_t)(i);
  }

  if (DEBUG) syslog(LOG_INFO,"creating %d threads (one per client)",NCLIENTS);
    
  for(int i=0; i<NCLIENTS; i++){
    if (pthread_create(&threads[i], &attr, &process, (void *)(&args[i]))) {
      syslog(LOG_ERR,"Failed to create thread %d\n", i);
    }
  }
  pthread_attr_destroy(&attr);
  if (DEBUG) syslog(LOG_INFO,"threads kinda running");
  
  int observation_complete=0;
  int blocks = 0;
  int ctt;
  int bswitch;
  
  syslog(LOG_INFO, "starting observation");

  while (!observation_complete) {

    // look for complete block

    //if (DEBUG) syslog(LOG_INFO,"here with %d",blockct);
    usleep(10);

    if (blockct>=NCLIENTS*NSAMPS_PER_BLOCK/NSAMPS_PER_TRANSMIT) {      
      
      // change output
      bswitch= block_switch;
      blockct=0;
      if (bswitch==0) block_switch=1;
      if (bswitch==1) block_switch=0;

      // write to output
      if (bswitch==0) written = ipcio_write (hdu_out->data_block, output1, block_out);
      if (bswitch==1) written = ipcio_write (hdu_out->data_block, output2, block_out);
      if (written < block_out)
	{
	  syslog(LOG_ERR, "main: failed to write all data to datablock [output]");	
	  dsaX_dbgpu_cleanup (hdu_out);
	  return EXIT_FAILURE;
	}

      if (DEBUG) syslog(LOG_INFO, "written block %d",blocks);      
      blocks++;
      ctt=0;
    }
      
  }
  
  // free stuff
  for(int i=0; i<NCLIENTS; i++){
    pthread_join(threads[i], &result);
    if (DEBUG) syslog(LOG_INFO,"joined thread %d",i);
  }
  free(output1);
  free(output2);
  dsaX_dbgpu_cleanup(hdu_out);
  
}
