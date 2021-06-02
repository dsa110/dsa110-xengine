/* This works pretty much like the trigger code. receives a control UDP message 
to store some data for a fixed amount of time.
Message format: length(s)-NAME
Will ignore messages until data recording is over
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/syscall.h>
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
  float fch1 = 1530.0;
  char fnam[300], foutnam[400];
  sprintf(fnam,"/home/dsa/alltest");

  // for getting MJD
  FILE *fmjd;
  int get_mjd = 0;
  int sumi=1;
  
  while ((arg=getopt(argc,argv,"c:f:o:i:k:s:mdh")) != -1)
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
  float *hoblock = (float *)malloc(sizeof(float)*64*1024*16384/sumi);  
  
  // start things

  syslog(LOG_INFO, "dsaX_writespec: starting observation");
  int nblocks = 0;
  
  while (!observation_complete) {

    // read block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    if (DEBUG) for (int i=0;i<48;i++) syslog(LOG_INFO,"%hu",((unsigned char *)(block))[i]);

    for (int i=0;i<64*1024*16384/sumi;i++) hoblock[i] = 0.;
    
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
	//NINTS = (int)(floor(reclen/(0.134217728)));
	sprintf(foutnam,"%s_%s_%d_%d.fil",fnam,srcnam,fctr,nblocks);
	syslog(LOG_INFO, "main: opening new file %s",foutnam);

	if (!(output = fopen(foutnam,"wb"))) {
	  printf("Couldn't open output file\n");
	  return 0;	  
	}

	if (get_mjd==1) {
	  if (!(fmjd = fopen("/home/ubuntu/tmp/mjd.dat","r"))) {
	    syslog(LOG_ERR,"could not open fmjd");
	  }
	  fscanf(fmjd,"%lf",&mjd);
	  mjd += nblocks*4.294967296/86400.;
	  fclose(fmjd);
	}
	  

	send_string("HEADER_START");
	send_string("source_name");
	send_string(srcnam);
	send_int("machine_id",1);
	send_int("telescope_id",82);
	send_int("data_type",1); // filterbank data
	send_double("fch1",1530.0); // THIS IS CHANNEL 0 :)
	send_double("foff",-0.244140625);
	send_int("nchans",1024);
	if (sumi==1) send_int("nbits",8);
	else send_int("nbits",32);	
	send_double("tstart",mjd);
	send_double("tsamp",8.192e-6*8.*4.*sumi);
	send_int("nifs",1);
	send_string("HEADER_END");
	
	syslog(LOG_INFO, "main: opened new file %s",foutnam);
		
	dfwrite=1;

	
      }      
      
      // write data to file
      syslog(LOG_INFO,"writing");

      
      for (int i=0;i<64;i++) {
	for (int j=0;j<16384/sumi;j++) {
	  for (int k=0;k<sumi;k++) {
	    for (int l=0;l<1024;l++) {
	      hoblock[i*16384*1024/sumi + j*1024 + l] += 1.*((unsigned char *)(block))[i*16384*1024 + (j*sumi+k)*1024 + l];
	    }
	  }
	}
      }
	      
      
      if (sumi==1) fwrite((unsigned char *)(block),sizeof(unsigned char),block_size,output);
      else fwrite(hoblock,sizeof(float),block_size/sumi/64,output);
      //fwrite(oblock,sizeof(float),256*48,output);

      integration++;
      // check if file writing is done
      if (integration==NINTS) {
	fclose(output);
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
  void* result=0;
  pthread_join (control_thread_id, &result);

  free(hoblock);
  dsaX_dbgpu_cleanup(hdu_in);
 
}
