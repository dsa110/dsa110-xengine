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
#include "fitsio.h"
#include "xgpu.h"

/* global variables */
int quit_threads = 0;
int dump_pending = 0;
int trignum = 0;
int dumpnum = 0;
char iP[100];
char srcnam[1024];
float reclen;
int DEBUG = 0;

// assumes that only first 6 baselines are written and 384 channels and 2 pols
const int n = 9216;
float summed_vis[9216];
const int n_all = 3194880;

// for extracting data
// assumes TRIANGULAR_ORDER for mat (f, baseline, pol, ri)
void simple_extract(Complex *mat, float *output);

void simple_extract(Complex *mat, float *output) {

  int in_idx, out_idx;
  for (int bctr=0;bctr<2080;bctr++) {
    for (int pol1=0;pol1<2;pol1++) {

      for (int f=0;f<384;f++) {

	out_idx = 2*((bctr*384+f)*2+pol1);
	in_idx = (2*f*2080+bctr)*4+pol1*3;
	output[out_idx] = 0.5*(mat[in_idx].real + mat[in_idx+8320].real);
	output[out_idx+1] = 0.5*(mat[in_idx].imag + mat[in_idx+8320].imag);

      }
    }
  }

}




void dsaX_dbgpu_cleanup (dada_hdu_t * in);

void usage()
{
  fprintf (stdout,
	   "dsaX_image [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -d debug [default no]\n"
	   " -f filename base [default test.fits]\n"
	   " -o freq of chan 1 [default 1494.84375]\n"
	   " -i IP to listen to [no default]\n"
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
  openlog ("dsaX_writevis", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA defs */
  dada_hdu_t* hdu_in = 0;
  multilog_t* log = 0;
  key_t in_key = XGPU_BLOCK_KEY;

  /* actual struct with info */
  udpdb_t udpdb;
  
  // command line
  int arg = 0;
  int core = -1;
  float fch1 = 1500.0;
  int nchans = 384;
  char fnam[300], foutnam[400];
  sprintf(fnam,"/home/dsa/alltest");
  
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
	      printf ("ERROR: -c flag requires argument\n");
	      return EXIT_FAILURE;
	    }
	case 'f':
	  strcpy(fnam,optarg);
	  break;
	case 'd':
	  DEBUG=1;
	  break;
	case 'o':
	  fch1 = atof(optarg);
	  break;
	case 'i':
	  strcpy(iP,optarg);
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // DADA stuff
  
  udpdb.verbose = 1;

  syslog (LOG_INFO, "dsaX_writevis: creating hdu");

  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"dsaX_writevis: could not connect to dada buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"dsaX_writevis: could not lock to dada buffer");
    return EXIT_FAILURE;
  }

  // Bind to cpu core
  if (core >= 0)
    {
      syslog(LOG_INFO,"binding to core %d", core);
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"dsaX_writevis: failed to bind to core %d", core);
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
  fitsfile *fptr;
  int rownum = 1;
  int fwrite = 0;
  int status=0;
  float mytsamp = 4096*4*8.192e-6;
  int NINTS;
  
  // data stuff
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t bytes_read = 0, block_id;
  char *block;
  float *data = (float *)malloc(sizeof(float)*n_all);
  int si1, si2;
  int nblocks = 0;
  Complex * cblock; 
  
  // start things

  syslog(LOG_INFO, "dsaX_writevis: starting observation");

  while (!observation_complete) {

    // read block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    cblock = (Complex *)(block);

    //if (DEBUG) {
      if (nblocks==20) {
	for (int i=100;i<200;i++) {
	  syslog(LOG_DEBUG,"MAT %d %f %f",i,(float)(cblock[i].real),(float)(cblock[i].imag));
	}
      }
      //}
    
    // DO STUFF - from block to summed_vis

    if (DEBUG) syslog(LOG_DEBUG,"extracting...");
    simple_extract((Complex *)(block), data);
    for (int i=0;i<n;i++) summed_vis[i] = data[i];
    if (DEBUG) syslog(LOG_DEBUG,"extracted!");
    
    // check for dump_pending
    if (dump_pending) {

      // if file writing hasn't started
      if (fwrite==0) {

	syslog(LOG_INFO, "dsaX_writevis: beginning file write for SRC %s for %f s",srcnam,reclen);
	status=0;
	
	NINTS = (int)(floor(reclen/mytsamp));
	sprintf(foutnam,"%s_%s_%d.fits",fnam,srcnam,fctr);
	syslog(LOG_INFO, "main: opening new file %s",foutnam);
	rownum=1;
	
	char *ttype[] = {"VIS"};
	char *tform[] = {"9216E"}; // assumes classic npts
	char *tunit[] = {"\0"};
	char *wsrcnam = srcnam;
	
	char extname[] = "DATA";
	fits_create_file(&fptr, foutnam, &status);
	if (status) syslog(LOG_ERR, "create_file FITS error %d",status);
	fits_create_tbl(fptr, BINARY_TBL, 0, 1, ttype, tform, tunit, extname, &status);
	fits_write_key(fptr, TFLOAT, "TSAMP", &mytsamp, "Sample time (s)", &status);
	fits_write_key(fptr, TFLOAT, "FCH1", &fch1, "Frequency (MHz)", &status);
	fits_write_key(fptr, TINT, "NCHAN", &nchans, "Channels", &status);
	fits_write_key(fptr, TSTRING, "Source", &wsrcnam[0], "Source", &status);	  
	fits_write_key(fptr, TINT, "NBLOCKS", &nblocks, "Ints", &status);
	if (status) syslog(LOG_ERR, "fits_write FITS error %d",status);
	fits_close_file(fptr, &status);

	fwrite=1;
	
      }

      // write data to file
      fits_open_table(&fptr, foutnam, READWRITE, &status);
      fits_write_col(fptr, TFLOAT, 1, rownum, 1, n, summed_vis, &status);
      rownum += 1;
      fits_update_key(fptr, TINT, "NAXIS2", &rownum, "", &status);
      fits_close_file(fptr, &status);
      integration++;
      if (status) syslog(LOG_ERR, "fits_write FITS error %d",status);	
      // check if file writing is done
      if (integration==NINTS) {
	integration=0;
	syslog(LOG_INFO, "dsaX_writevis: completed file %d",fctr);
	fctr++;
	fwrite=0;
	dump_pending=0;
      }

      syslog(LOG_INFO,"written");
      
    }
            
    // close off loop
    if (bytes_read < block_size)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    nblocks++;

    if (DEBUG) syslog(LOG_DEBUG,"Finished block %d",nblocks);
    
  }

  // close control thread
  syslog(LOG_INFO, "joining control_thread");
  quit_threads = 1;
  void* result=0;
  pthread_join (control_thread_id, &result);

  free(data);
  dsaX_dbgpu_cleanup(hdu_in);
 
}
