/* This works pretty much like the trigger code. receives a control UDP message 
to store some data for a fixed amount of time.
Message format: length(s)-NAME
Will ignore messages until data recording is over
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
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_capture.h"
#include "dsaX_def.h"

#include <src/sigproc.h>
#include <src/header.h>

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


/* global variables */
int quit_threads = 0;
int dump_pending = 0;
int trignum = 0;
int dumpnum = 0;
char iP[100];
char srcnam[1024];
float reclen;
int DEBUG = 0;

void dsaX_dbgpu_cleanup (dada_hdu_t * in);
void convert_block(char * b1, char * b2);

void usage()
{
  fprintf (stdout,
	   "dsaX_image [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -b write one beam\n"
	   " -f filename base [default test.fil]\n"
	   " -k in_key [BF_BLOCK_KEY]\n"
	   " -i IP to listen to [no default]\n"
	   " -s integrate N ints MUST BE FACTOR OF 16384 [default 1]\n"
	   " -m get mjd from file\n"
	   " -d DEBUG\n"
	   " -h        print usage\n");
}

void dsaX_dbgpu_cleanup (dada_hdu_t * in) {

  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);

}

// data to pass to threads
struct tdata {
  unsigned char * data;
  uint64_t block_size;
  char * filename;
  int n_threads;
  int thread_id;
};
int cores[4] = {31,32,33,34};
int NTHREADS = 4;

// thread to write out filterbank
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

  char myname[400];
  sprintf(myname,"%s_%d.fil",d->filename,thread_id);

  // DO STUFF
  syslog(LOG_INFO,"thread %d: writing to %s",thread_id,myname);
  FILE *moutput;
  moutput = fopen(myname,"ab");
  fwrite(d->data,sizeof(unsigned char),d->block_size,moutput);
  fclose(moutput);
  syslog(LOG_INFO,"thread %d: written",thread_id);
  
  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);



}



// Thread to control the dumping of data

void control_thread (void * arg) {

  udpdb_t * ctx = (udpdb_t *) arg;
  syslog(LOG_INFO, "control_thread: starting");

  // port on which to listen for control commands
  int port = WRITEVIS_CONTROL_PORT;
  char sport[10];
  sprintf(sport,"%d",port);
  
  // buffer for incoming command strings, and setup of socket
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
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
  getaddrinfo(iP,sport,&hints,&res);
  int fd;
  ssize_t ct;
  char tmpstr;
  char cmpstr = 'p';
  char *endptr;
  float tmp_reclen;
  
  syslog(LOG_INFO, "control_thread: created socket on port %d", port);
  
  while (!quit_threads) {
    
    fd = socket(res->ai_family,res->ai_socktype,res->ai_protocol);
    bind(fd,res->ai_addr,res->ai_addrlen);
    memset(buffer,'\0',sizeof(buffer));
    syslog(LOG_INFO, "control_thread: waiting for packet");
    ct = recvfrom(fd,buffer,1024,0,(struct sockaddr*)&src_addr,&src_addr_len);
    
    syslog(LOG_INFO, "control_thread: received buffer string %s",buffer);
    trignum++;

    // interpret buffer string
    char * rest = buffer;
    tmp_reclen = (float)(strtof(strtok(rest, "-"),&endptr));
    char * tmp_srcnam = strtok(NULL, "-");
    
    if (!dump_pending) {
      reclen = tmp_reclen;
      strcpy(srcnam,tmp_srcnam);
      syslog(LOG_INFO, "control_thread: received command to dump %f s for SRC %s",reclen,srcnam);
    }
	
    if (dump_pending)
      syslog(LOG_ERR, "control_thread: BACKED UP - CANNOT dump %f s for SRC %s",tmp_reclen,tmp_srcnam);
  
    if (!dump_pending) dump_pending = 1;
    
    close(fd);
    
  }

  free (buffer);

  if (ctx->verbose)
    syslog(LOG_INFO, "control_thread: exiting");

  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);

}

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_writeFil", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA defs */
  dada_hdu_t* hdu_in = 0;
  multilog_t* log = 0;
  key_t in_key = BF_BLOCK_KEY;

  /* actual struct with info */
  udpdb_t udpdb;
  
  // command line
  int arg = 0;
  int core = -1;
  float fch1 = 1498.75;
  char fnam[300], foutnam[400], myoutnam[400];
  sprintf(fnam,"/home/dsa/alltest");

  // for getting MJD
  FILE *fmjd;
  int get_mjd = 0;
  int sumi=1;
  int onebeam=0;
  
  while ((arg=getopt(argc,argv,"c:f:o:i:k:s:bmdh")) != -1)
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
	      printf ("ERROR: -c flag requires argument\n");
	      return EXIT_FAILURE;
	    }
	case 'k':
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
	      syslog(LOG_ERR,"-k flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'f':
	  strcpy(fnam,optarg);
	  break;
	case 'i':
	  strcpy(iP,optarg);
	  break;
	case 'd':
	  DEBUG=1;
	  break;
	case 'b':
	  onebeam=1;
	  break;
	case 'm':
	  get_mjd=1;
	  break;
	case 's':
	  sumi = atoi(optarg);
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // DADA stuff
  
  udpdb.verbose = 1;

  syslog (LOG_INFO, "dsaX_writefil: creating hdu");

  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"dsaX_writefil: could not connect to dada buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"dsaX_writespec: could not lock to dada buffer");
    return EXIT_FAILURE;
  }

  // Bind to cpu core
  if (core >= 0)
    {
      syslog(LOG_INFO,"binding to core %d", core);
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"dsaX_writefil: failed to bind to core %d", core);
    }

  int observation_complete=0;

  // more DADA stuff - deal with headers
  
  uint64_t header_size = 0;

  // read the headers from the input HDUs and mark as cleared
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "main: could not read next header");
      dsaX_dbgpu_cleanup (hdu_in);
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in);
      return EXIT_FAILURE;
    }


  // start control thread
  int rval = 0;
  pthread_t control_thread_id;
  syslog(LOG_INFO, "starting control_thread()");
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &udpdb);
  if (rval != 0) {
    syslog(LOG_INFO, "Error creating control_thread: %s", strerror(rval));
    return -1;
  }

  // set up threads
  struct tdata args[4];
  pthread_t threads[4];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  void* result=0;

  // set up
  int fctr = 0, integration = 0;
  char tstamp[100];
  double mjd=55000.;
  int rownum = 1;
  int dfwrite = 0;
  float mytsamp = 4.*8.*8.192e-6;
  int NINTS, midx;
  
  // data stuff
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t bytes_read = 0, block_id;
  char *block;
  //float *hoblock = (float *)malloc(sizeof(float)*128*768*16384/sumi);  
  
  // start things

  syslog(LOG_INFO, "dsaX_writespec: starting observation");
  int nblocks = 0;
  
  while (!observation_complete) {

    // read block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    if (DEBUG) for (int i=0;i<48;i++) syslog(LOG_INFO,"%hu",((unsigned char *)(block))[i]);

    //for (int i=0;i<128*768*16384/sumi;i++) hoblock[i] = 0.;
    
    // for writing sum
    /*    for (int i=0;i<256*48;i++) oblock[i] = 0.;
    for (int i=0;i<128;i++) {
      for (int j=0;j<256*48;j++) oblock[j] += (float)(block[i*256*48+j]);
      }*/
    
    syslog(LOG_INFO,"read block %d",nblocks);
        
    // check for dump_pending
    if (dump_pending) {

      // if file writing hasn't started
      if (dfwrite==0) {

	syslog(LOG_INFO, "beginning file write for SRC %s for %f s",srcnam,reclen);
	NINTS = (int)(floor(reclen/(mytsamp*16384.)));

	if (get_mjd==1) {
	  if (!(fmjd = fopen("/home/ubuntu/tmp/mjd.dat","r"))) {
	    syslog(LOG_ERR,"could not open fmjd");
	  }
	  fscanf(fmjd,"%lf",&mjd);
	  mjd += nblocks*4.294967296/86400.;
	  fclose(fmjd);
	}

	// set up each file

	for (int ith=0;ith<NTHREADS;ith++) {
	
	  sprintf(foutnam,"%s_%s_%d_%d_",fnam,srcnam,fctr,nblocks,ith);
	  syslog(LOG_INFO, "main: opening new file like %s",foutnam);

	  args[ith].block_size = block_size / 4;
	  args[ith].filename = foutnam;
	  args[ith].n_threads = 4;
	  args[ith].thread_id = ith;

	  sprintf(myoutnam,"%s_%d.fil",foutnam,ith);
	  if (!(output = fopen(myoutnam,"wb"))) {
	    printf("Couldn't open output file\n");
	    return 0;	  
	  }

	  send_string("HEADER_START");
	  send_string("source_name");
	  send_string(srcnam);
	  send_int("machine_id",1);
	  send_int("telescope_id",82);
	  send_int("data_type",1); // filterbank data
	  send_double("fch1",1498.75); // THIS IS CHANNEL 0 :)
	  send_double("foff",-0.244140625);
	  send_int("nchans",768);
	  send_int("nbits",8);
	  send_double("tstart",mjd);
	  send_double("tsamp",8.192e-6*8.*4.);
	  send_int("nifs",1);
	  send_string("HEADER_END");
	  
	  syslog(LOG_INFO, "main: opened new file %s",foutnam);

	  fclose(output);

	}
		
	dfwrite=1;

      }
      
      // write data to file
      syslog(LOG_INFO,"writing");

      syslog(LOG_INFO, "creating threads");
      
      for(int i=0; i<4; i++){
	args[i].data = (unsigned char *)(block) + i*32*16384*768;
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

      

      integration++;
      // check if file writing is done
      if (integration==NINTS) {
	integration=0;
	syslog(LOG_INFO, "dsaX_writespec: completed file %d",fctr);
	fctr++;
	dfwrite=0;
	dump_pending=0;
      }

      syslog(LOG_INFO,"written");
      
    }
            
    // close off loop
    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    nblocks += 1;
    
  }

  // close control thread
  syslog(LOG_INFO, "joining control_thread");
  quit_threads = 1;
  pthread_join (control_thread_id, &result);

  dsaX_dbgpu_cleanup(hdu_in);
 
}
