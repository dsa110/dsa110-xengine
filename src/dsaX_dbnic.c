/* simple nicdb

will work on NBMS/NBEAMS_PER_BLOCK writers, ip addresses set in code for now  

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


// data to pass to threads
struct data {
  char * out;
  int sockfd; 
  int thread_id;
  int chgroup;
  int tseq;
};

/* global variables */
int DEBUG = 0;
int TEST = 0;

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
	   "dsaX_dbnic [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -g chgroup [default 0]\n"
	   " -d send debug messages to syslog\n"
	   " -t TEST\n"
	   " -i in_key [default BF_BLOCK_KEY]\n"
	   " -w -x -y -z four ip addresses for corner turn\n"
	   " -h print usage\n");
}

/* thread for data transmission */
void * transmit(void *args) {

  // basic stuff
  struct data *d = args;
  int thread_id = d->thread_id;
  int sockfd = d->sockfd; 
  char * output = (char *)(d->out);
  char * op = (char *)malloc(sizeof(char)*(8+NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW));
  int * iop = (int *)(op);
  int chgroup = d->chgroup;
  int tseq = d->tseq;

  // fill op, doing transpose
  iop[0] = chgroup;
  iop[1] = tseq;
  for (int i=0;i<NSAMPS_PER_TRANSMIT;i++) {
    for (int j=0;j<NBEAMS_PER_BLOCK;j++) {
      for (int k=0;k<NW;k++) 
	// op[8+i*NBEAMS_PER_BLOCK*NW+j*NW+k] = output[i*NBMS*NW + thread_id*NBEAMS_PER_BLOCK*NW + j*NW+k]; // no transpose
	op[8+j*NSAMPS_PER_TRANSMIT*NW+i*NW+k] = output[i*NBMS*NW + thread_id*NBEAMS_PER_BLOCK*NW + j*NW+k]; // yes transpose
    }
  }

  if (DEBUG) syslog(LOG_DEBUG,"sending with chgroup %d tseq %d",iop[0],iop[1]);
  
  // do transmit
  int remain_data = (int)((8+NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW));
  int sent_bytes = 0, sbytes;
  while (((sbytes = send(sockfd, op + sent_bytes, remain_data, 0))>0) && (remain_data > 0)) {
    remain_data -= sbytes;
    sent_bytes += sbytes;
  }
  


  //  write(sockfd, op, sizeof(op));

  if (DEBUG) syslog(LOG_DEBUG,"thread %d: written output",thread_id);
  
  /* return 0 */
  free(op);
  int thread_result = 0;
  pthread_exit((void *) &thread_result);
  
}


// MAIN

int main (int argc, char *argv[]) {
  
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_dbnic", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  // threads
  struct data args[4];
  pthread_t threads[4];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  
  // command line arguments
  int core = -1;
  int chgroup = 0;
  int arg = 0;
  char iP[4][20] = {"10.41.0.22", "10.41.0.98", "10.41.0.105", "10.41.0.63"}; 
  // data block HDU keys
  key_t in_key;
  in_key = BF_BLOCK_KEY;

  
  while ((arg=getopt(argc,argv,"c:g:ti:w:x:y:z:dh")) != -1)
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
	case 'g':
	  if (optarg)
	    {
	      chgroup = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-c flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'w':
	  strcpy(iP[0],optarg);
	  break;
	case 'x':
	  strcpy(iP[1],optarg);
	  break;
	case 'y':
	  strcpy(iP[2],optarg);
	  break;
	case 'z':
	  strcpy(iP[3],optarg);
	  break;
	case 't':
	  TEST=1;
	  syslog (LOG_INFO, "Will use test pattern");
	  break;
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
  syslog(LOG_INFO, "main: have input and output block sizes %llu",block_size);
  uint64_t  bytes_read = 0;
  char *block;
  uint64_t written, block_id;

  
  // set up
  int observation_complete=0;
  int blocks = 0;
  int started = 0;
  int nthreads = NBMS / NBEAMS_PER_BLOCK;
  
  
  // create socket connections
  int sockfd[nthreads];
  struct sockaddr_in servaddr;
  for (int i=0;i<nthreads;i++) sockfd[i] = socket(AF_INET, SOCK_STREAM, 0);
  if (DEBUG) syslog(LOG_DEBUG,"sockets created");
  for (int i=0;i<nthreads;i++) {
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(iP[i]);
    servaddr.sin_port = htons(FIL_PORT0+(uint16_t)(chgroup));
    if (connect(sockfd[i], (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
      syslog(LOG_ERR,"connection with the server failed %d",i);
      exit(0);
    }
    if (DEBUG) syslog(LOG_DEBUG,"connected %d",i);
  }
  
  syslog(LOG_INFO, "starting observation");

  /*
  block has size/shape [NSAMPS_PER_TRANSMIT, NBMS, NW]
  want to transmit [NBEAMS_PER_BLOCK, NSAMPS_PER_TRANSMIT, NW]
  for test tone, populate with chgroup*10 + beam*NBMS/NBEAMS_PER_BLOCK + time*2/NSAMPS_PER_TRANSMIT
  */
  unsigned char * testblock = (unsigned char *)malloc(sizeof(unsigned char)*block_size);
  for (int i=0;i<NSAMPS_PER_TRANSMIT;i++) {
    for (int j=0;j<NBMS;j++) {
      for (int k=0;k<NW;k++)
	testblock[i*NBMS*NW + j*NW + k] = (unsigned char)(i/2);
    }
  }
  
  
  while (!observation_complete) {

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    if (started==0) {
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }

    // DO STUFF

    //if (DEBUG) syslog(LOG_DEBUG,"creating %d threads",nthreads);

    // put together args
    for (int i=0; i<nthreads; i++) {
      if (TEST) args[i].out = testblock;
      else args[i].out = block;
      args[i].sockfd = sockfd[i];
      args[i].thread_id = i;
      args[i].chgroup = chgroup;
      args[i].tseq = blocks;
    }
    
    for(int i=0; i<nthreads; i++){
      if (pthread_create(&threads[i], &attr, &transmit, (void *)(&args[i]))) {
	syslog(LOG_ERR,"Failed to create massage thread %d\n", i);
      }
    }

    pthread_attr_destroy(&attr);
    //if (DEBUG) syslog(LOG_DEBUG,"threads kinda running");
    
    for(int i=0; i<nthreads; i++){
      pthread_join(threads[i], &result);
      //if (DEBUG) syslog(LOG_DEBUG,"joined thread %d",i);
    }

    if (DEBUG) syslog(LOG_DEBUG, "written block %d",blocks);      
    blocks++;
    

    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  for (int i=0;i<nthreads;i++) close(sockfd[i]);
  free(testblock);
  dsaX_dbgpu_cleanup (hdu_in);
  
}


