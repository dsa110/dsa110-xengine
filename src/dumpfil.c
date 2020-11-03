//E_GNU
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

// global variables
int DEBUG = 0;

void usage()
{
  fprintf (stdout,
	   "dumpfil [options]\n"
	   " -d send debug messages to syslog\n"
	   " -p no header\n"
	   " -f file to dump to [default none]\n"
	   " -n blocks to dump [default 30]\n"
	   " -i in_key [default TEST_BLOCK_KEY]\n"
	   " -h print usage\n");
}


void dsaX_dbgpu_cleanup (dada_hdu_t * in);


void dsaX_dbgpu_cleanup (dada_hdu_t * in)
{

  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);
  
}

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



// MAIN

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dumpfil", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;

  // data block HDU keys
  key_t in_key = 0x0000aaae;
  
  // command line arguments
  char fnam[100];
  sprintf(fnam,"/home/ubuntu/dumpfil.fil");
  int nbl = 30;
  int arg = 0;
  int nhd = 0;
  
  while ((arg=getopt(argc,argv,"f:i:n:pdh")) != -1)
    {
      switch (arg)
	{
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
	      strcpy(fnam,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'n':
	  if (optarg)
	    {
	      nbl = atoi(optarg);	      
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-n flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'p':
	  nhd=1;
	  syslog (LOG_INFO, "Will not write a header");
	  break;
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  syslog(LOG_INFO,"will use %d blocks",nbl);
  
  // DADA stuff
  
  syslog (LOG_INFO, "creating in hdus");
  
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
  syslog(LOG_INFO, "main: have input block size %llu\n",block_size);
  uint64_t  bytes_read = 0;
  uint64_t npackets = 1;
  char * block, * output_buffer;
  uint64_t written, block_id;

  // fill output buffer if file exists
  output=fopen(fnam,"wb");
  if(output == NULL)
    {
      syslog(LOG_ERR,"Error opening file");
      exit(1);
    }

  if (!nhd) {
    send_string("HEADER_START");
    send_string("source_name");
    send_string("TESTSRC");
    send_int("machine_id",1);
    send_int("telescope_id",82);
    send_int("data_type",1); // filterbank data
    send_double("fch1",1530.0); // THIS IS CHANNEL 0 :)
    send_double("foff",-0.244140625);
    send_int("nchans",48);
    send_int("nbits",8);
    send_double("tstart",55000.0);
    send_double("tsamp",8.192e-6*8.*16.);
    send_int("nifs",1);
    send_string("HEADER_END");
  }
  
  int observation_complete=0;
  int blocks = 0, started = 0;
  
  syslog(LOG_INFO, "starting observation");


  while (blocks < nbl) {

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    fwrite(block, sizeof(char), bytes_read, output);
    blocks++;
    
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    
  }

  fclose(output);
  dsaX_dbgpu_cleanup (hdu_in);
  
}
