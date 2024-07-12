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

#define bdepth 16
#define MAX_FULLBLOCK 4

// global variables
int DEBUG = 0;
volatile int blockct[bdepth]; // to count how many writes to block. max is NSAMPS_PER_BLOCK*NBEAMS_PER_BLOCK*NW
volatile int flush_flag = 0; // set to flush output2
volatile int writing = 0;
volatile int global_tseq = 0; // global count of full buffers
int cores[16] = {3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28}; // to bind threads to
char iP[100];
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;	  

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
  struct sockaddr_in si_other, si_me;
  int clientSocket, slen=sizeof(si_other);
  clientSocket=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (DEBUG) syslog(LOG_INFO,"thread %d: Made socket",thread_id);
  memset((char *) &si_me, 0, sizeof(si_me));
  si_me.sin_family = AF_INET;
  si_me.sin_port = htons(tport);
  si_me.sin_addr.s_addr = inet_addr(iP);
  if (bind(clientSocket, (struct sockaddr *)&si_me, sizeof(si_me)) < 0) {
    syslog(LOG_ERR,"thread %d: cannot bind to port",thread_id);
    exit(1);
  }
  if (DEBUG) syslog(LOG_INFO,"thread %d: socket bound - waiting for header packet",thread_id);

  char * packet = (char *)malloc(sizeof(char)*P_SIZE);
  int * ibuf;
  recvfrom(clientSocket, packet, P_SIZE, 0,(struct sockaddr *)&si_other,&slen);
  ibuf = (int *)(packet);
  int chgroup = ibuf[0];
  syslog(LOG_INFO,"thread %d: accepted connection from chgroup %d",thread_id,chgroup);

  // data buffer and other variables
  char * buffer = (char *)malloc((NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW)*sizeof(char));
  int tseq, pseq;
  int pct = 0;
  int full_blocks = 0;
  int fullBlock;
  int i0, aa;
  int lastPacket, nextBuf, current_tseq = 0, act_tseq; 
  uint64_t shifty = (bdepth-1)*NSAMPS_PER_BLOCK*NBEAMS_PER_BLOCK*NCHAN_FIL;
  uint64_t oidx_offset, oidx;
  
  // infinite loop 
  while (1) {
  
    /* read message */
    // fill up local buffer
    lastPacket = 0;
    nextBuf = 0;
    while ((lastPacket==0) && (nextBuf==0)) {

      recvfrom(clientSocket, packet, P_SIZE, 0,(struct sockaddr *)&si_other,&slen);
      ibuf = (int *)(packet);
      pseq = ibuf[2];
      if (chgroup != ibuf[0]) 
	syslog(LOG_ERR,"thread %d: received chgroup %d is not recorded %d",thread_id,ibuf[0],chgroup);
      tseq = ibuf[1];

      if (tseq>current_tseq) {
	nextBuf=1;
      }
      else if (tseq==current_tseq) {
	memcpy(buffer+pseq*(P_SIZE-12),packet+12,P_SIZE-12);
	pct++;
      }

      if (pseq==NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW/(P_SIZE-12)-1)
	lastPacket=1;

    }
    
    if (pct != NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW/(P_SIZE-12))
      syslog(LOG_ERR,"thread %d: only received %d of %d",thread_id,pct,NSAMPS_PER_TRANSMIT*NBEAMS_PER_BLOCK*NW/(P_SIZE-12));
    
    act_tseq = (current_tseq * NSAMPS_PER_TRANSMIT) % NSAMPS_PER_BLOCK; // place within output buffer

    // at this stage we have a full local buffer
    // this needs to be placed in the global buffer
      
    // output order is [beam, time, freq]. input order is [beam, time, freq], but only a subset of freqs
    i0 = 0;
    aa = ((current_tseq / (NSAMPS_PER_BLOCK/NSAMPS_PER_TRANSMIT)) % bdepth);
    oidx_offset = ((uint64_t)(aa))*NSAMPS_PER_BLOCK*NBEAMS_PER_BLOCK*NCHAN_FIL;
    //syslog(LOG_INFO,"thread %d: read message with chgroup %d tseq %d current_tseq %d global_tseq %d position %d %"PRIu64"",thread_id,chgroup,tseq,current_tseq,global_tseq,aa,oidx_offset);
    for (int i=0;i<NBEAMS_PER_BLOCK;i++) {
      for (int j=0;j<NSAMPS_PER_TRANSMIT;j++) {	
	for (int k=0;k<NW;k++) {
	  
	  oidx = oidx_offset + i*NSAMPS_PER_BLOCK*NCHAN_FIL + (act_tseq+j)*NCHAN_FIL + CHOFF/8 + chgroup*NW + k;
	  
	  output1[oidx] = buffer[i0];

	  i0++;
	    
	}
      }
    }
    //syslog(LOG_INFO,"thread %d: entering mutex",thread_id);

    // at this stage we have dealt with this capture round, and must address blockct within mutex
    pthread_mutex_lock(&mutex);

    // increment appropriate blockct
    aa = ((current_tseq / (NSAMPS_PER_BLOCK/NSAMPS_PER_TRANSMIT)) % bdepth);
    blockct[aa] += 1;
    //syslog(LOG_INFO,"thread %d: incrementing blockct %d %d %d (total %d)",thread_id,current_tseq,aa,blockct[aa],NCLIENTS*NSAMPS_PER_BLOCK/NSAMPS_PER_TRANSMIT);

    // deal with full block anywhere
    full_blocks=0;
    for (int i=0;i<bdepth;i++) {
      if (blockct[i]!=0) full_blocks++;
    }	
    for (int i=0;i<bdepth;i++) {
      if ((blockct[i] == NCLIENTS*NSAMPS_PER_BLOCK/NSAMPS_PER_TRANSMIT) || (full_blocks>=MAX_FULLBLOCK && blockct[i] >= (NCLIENTS-1)*NSAMPS_PER_BLOCK/NSAMPS_PER_TRANSMIT)) {

	// need to write this block and reset blockct
	while (flush_flag==1)
	  aa==1;
	flush_flag = 1;
	blockct[i] = 0;
	// log - hardcoded bdepth
	full_blocks -= 1;
	syslog(LOG_INFO,"thread %d: Writing global_tseq %d. Blockcts_full %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",thread_id,global_tseq,full_blocks,blockct[0],blockct[1],blockct[2],blockct[3],blockct[4],blockct[5],blockct[6],blockct[7],blockct[8],blockct[9],blockct[10],blockct[11],blockct[12],blockct[13],blockct[14],blockct[15]);

	
      }	

    }
        
    pthread_mutex_unlock(&mutex);

    // advance local tseq and deal with packet capture
    if (lastPacket==1) {
      current_tseq++;
      lastPacket=0;
      nextBuf=0;
      pct=0;
    }
    if (nextBuf==1) {
      current_tseq++;
      memcpy(buffer+pseq*(P_SIZE-12),packet+12,P_SIZE-12);
      pct=1;
      lastPacket=0;
    }

    

  }

  /* close socket and clean up */
  close(clientSocket);
  free(packet);
  free(buffer);
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
  for (int i=0;i<bdepth;i++) blockct[i] = 0;

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
		syslog(LOG_ERR, "could not parse key from %s", optarg);
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

  hdu_out  = dada_hdu_create (0);
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
  syslog(LOG_INFO, "main: have output block sizes %lu\n",block_out);
  uint64_t  bytes_read = 0;
  char *output1, *output2;
  output1 = (char *)malloc(sizeof(char)*block_out*bdepth);
  output2 = (char *)malloc(sizeof(char)*block_out);
  memset(output1,0,block_out*bdepth);
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
  int aa;
  
  syslog(LOG_INFO, "starting observation");

  while (!observation_complete) {

    // look for complete block

    //if (DEBUG) syslog(LOG_INFO,"here with %d",blockct);
    while (flush_flag==0)
      aa=1;

    // write to output
    writing=1;
    written = ipcio_write (hdu_out->data_block, output1 + (global_tseq % bdepth)*block_out, block_out);
    global_tseq += 1;
    writing=0;
    if (written < block_out)
      {
	syslog(LOG_ERR, "main: failed to write all data to datablock [output]");	
	dsaX_dbgpu_cleanup (hdu_out);
	return EXIT_FAILURE;
      }
    
    syslog(LOG_INFO, "written block %d",blocks);      
    blocks++;

    flush_flag = 0;

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
