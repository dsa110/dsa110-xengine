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

/* global variables */
int DEBUG = 0;
int STATS = 0;
const int nth = 4;

// data to pass to threads
struct data {
  char * in;
  char * out;
  char * out2;
  int bf;
  int reorder;
  int n_threads;
  int thread_id;
};
int cores[8] = {10, 11, 12, 13, 14, 15, 16, 17};


void * massage (void *args) {

  struct data *d = args;
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
  char *in = (char *)d->in;
  int bf = d->bf;
  int reorder = d->reorder;
  int n_threads = d->n_threads;  
  
  if (!reorder) {
    memcpy(d->out + thread_id*(2048/n_threads)*1536*NANT, in + thread_id*(2048/n_threads)*1536*NANT, (2048/n_threads)*1536*NANT);
    if (bf)
      memcpy(d->out2 + thread_id*(2048/n_threads)*1536*NANT, in + thread_id*(2048/n_threads)*1536*NANT, (2048/n_threads)*1536*NANT);
  }
  else {
  
    // block for transpose
    int block = 16;
  
    for (int i=(int)(thread_id*(2048/n_threads));i<(int)((thread_id + 1)*2048/n_threads);i++) { // over time
      for (int i1 = 0; i1 < 48; i1 += block) {
	for(int j = 0; j < NANT; j++) {
	  for(int b = 0; b < block && i1 + b < 48; b++) {
	    memcpy(d->out + i*1536*NANT + (i1+b)*NANT*32 + j*32, in + i*1536*NANT + j*1536 + (i1+b)*32, 32);
	    if (bf) memcpy(d->out2 + i*1536*NANT + (i1+b)*NANT*32 + j*32, in + i*1536*NANT + j*1536 + (i1+b)*32, 32);
	  }
	}
      }
    }    

  }
    
  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);
  
}


void dsaX_dbgpu_cleanup (dada_hdu_t * in, int write);
int dada_bind_thread_to_core (int core);
void reorder_block(char *block, char *output);
void calc_stats(char *block);

// calculates rms for each pol from the first packet in each block. 
// block has shape [2048 time, NANT antennas, 768 channels, 2 pol, r/i]
void calc_stats(char *input) {

  float rmss[NANT*2];
  int iidx;
  for (int i=0;i<NANT*2;i++) rmss[i] = 0.;

  for (int ant=0;ant<NANT;ant++) {
    for (int chan=0;chan<768;chan++) {
      for (int pol=0;pol<2;pol++) {

	iidx = ant*1536+chan*2+pol;
	
	rmss[ant*2+pol] += pow((float)(((char)((input[iidx] & 15) << 4)) >> 4),2.);
	rmss[ant*2+pol] += pow((float)(((char)((input[iidx] & 240))) >> 4),2.);

      }
    }
  }

  for (int i=0;i<NANT;i++) {
    if (STATS) syslog(LOG_INFO,"RMS_ant_2pol %d %g %g",i,sqrt(rmss[2*i]/768.),sqrt(rmss[2*i+1]/768.));
  }

}

// performs cpu reorder of block to be loaded to GPU
void reorder_block(char * block, char * output) {

  // from [2048 time, NANT antennas, 48 channels, 16 chunnels, 2 pol, r/i]
  // to [2048 time, 48 channels, NANT antennas, 16 chunnels, 2 pol, r/i]
  // 24576*NANT in total. 1536*NANT per time
  
  for (int i=0;i<2048;i++) { // over time
    for (int k=0;k<48;k++) { // over channels
      for (int j=0;j<NANT;j++) { // over ants
	// copy 32 bytes
	memcpy(output + i*1536*NANT + k*NANT*32 + j*32, block + i*1536*NANT + j*1536 + k*32, 32); 
	
      }
    }
  }

  //memcpy(block,output,24576*NANT);

}


void dsaX_dbgpu_cleanup (dada_hdu_t * in, int write)
{

  if (write==0) {
  
    if (dada_hdu_unlock_read (in) < 0)
      {
	syslog(LOG_ERR, "could not unlock read on hdu_in");
      }
    dada_hdu_destroy (in);

  }

  if (write==1) {

    if (dada_hdu_unlock_write (in) < 0)
      {
	syslog(LOG_ERR, "could not unlock write on hdu_in");
      }
    dada_hdu_destroy (in);

  }
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_split [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -m multithread write\n"
	   " -b connect to bf hdu\n"
	   " -r reorder\n"
	   " -i in_key [default CAPTURE_BLOCK_KEY]\n"
	   " -o out_key [default CAPTURED_BLOCK_KEY]\n"
	   " -j out_key2 [default REORDER_BLOCK_KEY2]\n"
	   " -s stats\n"
	   " -f send fake blocks through [default 0]\n"
	   " -h print usage\n");
}


// MAIN

int main (int argc, char *argv[]) {
  
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_split", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;
  dada_hdu_t* hdu_out2 = 0;

  // data block HDU keys
  key_t in_key = CAPTURE_BLOCK_KEY;
  key_t out_key = CAPTURED_BLOCK_KEY;
  key_t out_key2 = REORDER_BLOCK_KEY2;
  
  // command line arguments
  int core = -1;
  int bf = 0;
  int arg = 0;
  int reorder = 0;
  int mwrite = 0;
  int fake = 0;
  
  while ((arg=getopt(argc,argv,"c:i:o:j:f:smdbrh")) != -1)
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
	case 'f':
	  if (optarg)
	    {
	      fake = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
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
	case 'j':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &out_key2) != 1) {
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
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'r':
	  reorder=1;
	  syslog (LOG_INFO, "Will do reorder");
	  break;
	case 'm':
	  mwrite=1;
	  syslog (LOG_INFO, "Will do multithread write");
	  break;
	case 's':
	  STATS=1;
	  syslog (LOG_INFO, "Will print stats");
	  break;
	case 'b':
	  bf=1;
	  syslog (LOG_INFO, "Will write to bf dada hdu");
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

  if (bf) {
    hdu_out2  = dada_hdu_create ();
    dada_hdu_set_key (hdu_out2, out_key2);
    if (dada_hdu_connect (hdu_out2) < 0) {
      syslog (LOG_ERR,"could not connect to output  buffer2");
      return EXIT_FAILURE;
    }
    if (dada_hdu_lock_write(hdu_out2) < 0) {
      syslog (LOG_ERR, "could not lock to output buffer2");
      return EXIT_FAILURE;
    }
  }
  
  uint64_t header_size = 0;

  // deal with headers
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "could not read next header");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);
      
      
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
      return EXIT_FAILURE;
    }

  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in,0);
      dsaX_dbgpu_cleanup (hdu_out,1);
      if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
      return EXIT_FAILURE;
    }

  if (bf) {
    header_out = ipcbuf_get_next_write (hdu_out2->header_block);
    if (!header_out)
      {
	syslog(LOG_ERR, "could not get next header2 block [output]");
	dsaX_dbgpu_cleanup (hdu_in,0);
	dsaX_dbgpu_cleanup (hdu_out,1);
	if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);      
	return EXIT_FAILURE;
      }
    memcpy (header_out, header_in, header_size);
    if (ipcbuf_mark_filled (hdu_out2->header_block, header_size) < 0)
      {
	syslog (LOG_ERR, "could not mark header block2 filled [output]");
	dsaX_dbgpu_cleanup (hdu_in,0);
	dsaX_dbgpu_cleanup (hdu_out,1);
	if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);
	return EXIT_FAILURE;
      }
  }

  
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  uint64_t nints = block_size / block_out;
  syslog(LOG_INFO, "main: have input and output block sizes %llu %llu\n",block_size,block_out);
  uint64_t  bytes_read = 0;
  char * block, * output_buffer, * o1, * o2;
  output_buffer = (char *)malloc(sizeof(char)*block_out);
  char * output = (char *)malloc(sizeof(char)*block_out);
  memset(output_buffer,0,block_out);
  uint64_t written, block_id;

  // set up threads
  struct data args[8];
  pthread_t threads[8];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;
  
  // send through fake blocks

  if (fake>0) {
    syslog(LOG_INFO,"sending %d fake blocks",fake);
    for (int i=0;i<fake;i++) {
      o1 = ipcio_open_block_write (hdu_out->data_block, &block_id);
      memcpy(o1, output, block_out);
      ipcio_close_block_write (hdu_out->data_block, block_out);
      usleep(10000);
    }
    syslog(LOG_INFO,"Finished with fake blocks");
  }
  
  
  
  // set up

  int observation_complete=0;
  int blocks = 0;
  int started = 0;


  
  syslog(LOG_INFO, "starting observation");

  while (!observation_complete) {

    // open block
    
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    if (started==0) {
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }

    
    // DO STUFF

    for (int myint=0;myint<nints;myint++) {
        
      // copy to output buffer
                  
      memcpy(output_buffer, block + myint*block_out, block_out);      

      if (mwrite) {
	o1 = ipcio_open_block_write (hdu_out->data_block, &block_id);
	if (bf) o2 = ipcio_open_block_write (hdu_out2->data_block, &block_id);
      }
      
      // stats
      if (STATS) calc_stats(output_buffer);
      
      //if (reorder) {
      
      // set up data structure
      for (int i=0; i<nth; i++) {
	args[i].in = output_buffer;
	args[i].reorder = reorder;
	args[i].bf = 0;
	if (mwrite) {
	  args[i].out = o1;	
	  if (bf) {
	    args[i].out2 = o2;
	    args[i].bf = 1;
	  }
	}
	else
	  args[i].out = output;
	args[i].n_threads = nth;
	args[i].thread_id = i;
      }
      
      //if (DEBUG) syslog(LOG_DEBUG,"creating %d threads",nth);
      syslog(LOG_INFO, "creating threads");
      
      for(int i=0; i<nth; i++){
	if (pthread_create(&threads[i], &attr, &massage, (void *)(&args[i]))) {
	  syslog(LOG_ERR,"Failed to create massage thread %d\n", i);
	}
      }
      
      pthread_attr_destroy(&attr);
      if (DEBUG) syslog(LOG_DEBUG,"threads kinda running");
      
      for(int i=0; i<nth; i++){
	pthread_join(threads[i], &result);
	if (DEBUG) syslog(LOG_DEBUG,"joined thread %d",i);
      }
      
      
      if (!mwrite) {
	if (reorder && (!bf))
	  written = ipcio_write (hdu_out->data_block, output, block_out);
	else 
	  written = ipcio_write (hdu_out->data_block, output_buffer, block_out);
	
	if (bf) {
	  written = ipcio_write (hdu_out->data_block, output_buffer, block_out);
	  if (reorder)
	    written = ipcio_write (hdu_out2->data_block, output, block_out);
	  else
	    written = ipcio_write (hdu_out2->data_block, output_buffer, block_out);
	}
      }
      else {
	ipcio_close_block_write (hdu_out->data_block, block_out);
	if (bf) ipcio_close_block_write (hdu_out2->data_block, block_out);
      }
      
      if (DEBUG) syslog(LOG_DEBUG, "written block %d",blocks);      
      blocks++;
      
      
      if (bytes_read < block_size)
	observation_complete = 1;            
      
    }

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  free(output_buffer);
  free(output);
  dsaX_dbgpu_cleanup (hdu_in,0);
  dsaX_dbgpu_cleanup (hdu_out,1);
  if (bf) dsaX_dbgpu_cleanup (hdu_out2,1);	  
  
}


