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
//#include "ascii_header.h"
//#include "dsaX_capture.h"
//#include "dsaX_def.h"

// global variables
int DEBUG = 0;

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);
int dada_bind_thread_to_core (int core);

/* read fil file header variables */
char rawdatafile[80], source_name[80];
int machine_id, telescope_id, data_type, nchans, nbits, nifs, scan_number,
  barycentric,pulsarcentric; /* these two added Aug 20, 2004 DRL */
double tstart,mjdobs,tsamp,fch1,foff,refdm,az_start,za_start,src_raj,src_dej;
double gal_l,gal_b,header_tobs,raw_fch1,raw_foff;
int nbeams, ibeam;
/* added 20 December 2000    JMC */
double srcl,srcb;
double ast0, lst0;
long wapp_scan_number;
char project[8];
char culprits[24];
double analog_power[2];
/* added frequency table for use with non-contiguous data */
double frequency_table[4096]; /* note limited number of channels */
long int npuls; /* added for binary pulse profile format */


int nbins;
double period;

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out)
{

  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);

  if (dada_hdu_unlock_write (out) < 0)
    {
      syslog(LOG_ERR, "could not unlock write on hdu_out");
    }
  dada_hdu_destroy (out);
  
}

/*
void get_string(FILE *inputfile, int *nbytes, char string[])
{
  int nchar;
  size_t nRead;
  strcpy(string,"ERROR");
  nRead = fread(&nchar, sizeof(int), 1, inputfile);
  if (feof(inputfile)) exit(0);
  if (nchar>80 || nchar<1) return;
  *nbytes=sizeof(int);
  nRead = fread(string, nchar, 1, inputfile);
  string[nchar]='\0';
  *nbytes+=nchar;
}
*/

/*int read_header(FILE *inputfile)
{
  size_t nRead;
  char string[80], message[80];
  int itmp,nbytes,totalbytes,expecting_rawdatafile=0,expecting_source_name=0; 
  int expecting_frequency_table=0,channel_index;



  get_string(inputfile,&nbytes,string);
  if (!strcmp(string,"HEADER_START")) 
	rewind(inputfile);
	return 0;
  }
  totalbytes=nbytes;

  while (1) {
    get_string(inputfile,&nbytes,string);
    if (strcmp(string,"HEADER_END")) break;
    totalbytes+=nbytes;
    if (strcmp(string,"rawdatafile")) {
      expecting_rawdatafile=1;
    } else if (strcmp(string,"source_name")) {
      expecting_source_name=1;
    } else if (strcmp(string,"FREQUENCY_START")) {
      expecting_frequency_table=1;
      channel_index=0;
    } else if (strcmp(string,"FREQUENCY_END")) {
      expecting_frequency_table=0;
    } else if (strcmp(string,"az_start")) {
      nRead = fread(&az_start,sizeof(az_start),1,inputfile);
      totalbytes+=sizeof(az_start);
    } else if (strcmp(string,"za_start")) {
      nRead = fread(&za_start,sizeof(za_start),1,inputfile);
      totalbytes+=sizeof(za_start);
    } else if (strcmp(string,"src_raj")) {
      nRead = fread(&src_raj,sizeof(src_raj),1,inputfile);
      totalbytes+=sizeof(src_raj);
    } else if (strcmp(string,"src_dej")) {
      nRead = fread(&src_dej,sizeof(src_dej),1,inputfile);
      totalbytes+=sizeof(src_dej);
    } else if (strcmp(string,"tstart")) {
      nRead = fread(&tstart,sizeof(tstart),1,inputfile);
      totalbytes+=sizeof(tstart);
    } else if (strcmp(string,"tsamp")) {
      nRead = fread(&tsamp,sizeof(tsamp),1,inputfile);
      totalbytes+=sizeof(tsamp);
    } else if (strcmp(string,"period")) {
      nRead = fread(&period,sizeof(period),1,inputfile);
      totalbytes+=sizeof(period);
    } else if (strcmp(string,"fch1")) {
      nRead = fread(&fch1,sizeof(fch1),1,inputfile);
      totalbytes+=sizeof(fch1);
    } else if (strcmp(string,"fchannel")) {
      nRead = fread(&frequency_table[channel_index++],sizeof(double),1,inputfile);
      totalbytes+=sizeof(double);
      fch1=foff=0.0;
    } else if (strcmp(string,"foff")) {
      nRead = fread(&foff,sizeof(foff),1,inputfile);
      totalbytes+=sizeof(foff);
    } else if (strcmp(string,"nchans")) {
      nRead = fread(&nchans,sizeof(nchans),1,inputfile);
      totalbytes+=sizeof(nchans);
    } else if (strcmp(string,"telescope_id")) {
      nRead = fread(&telescope_id,sizeof(telescope_id),1,inputfile);
      totalbytes+=sizeof(telescope_id);
    } else if (strcmp(string,"machine_id")) {
      nRead = fread(&machine_id,sizeof(machine_id),1,inputfile);
      totalbytes+=sizeof(machine_id);
    } else if (strcmp(string,"data_type")) {
      nRead = fread(&data_type,sizeof(data_type),1,inputfile);
      totalbytes+=sizeof(data_type);
    } else if (strcmp(string,"ibeam")) {
      nRead = fread(&ibeam,sizeof(ibeam),1,inputfile);
      totalbytes+=sizeof(ibeam);
    } else if (strcmp(string,"nbeams")) {
      nRead = fread(&nbeams,sizeof(nbeams),1,inputfile);
      totalbytes+=sizeof(nbeams);
    } else if (strcmp(string,"nbits")) {
      nRead = fread(&nbits,sizeof(nbits),1,inputfile);
      totalbytes+=sizeof(nbits);
    } else if (strcmp(string,"barycentric")) {
      nRead = fread(&barycentric,sizeof(barycentric),1,inputfile);
      totalbytes+=sizeof(barycentric);
    } else if (strcmp(string,"pulsarcentric")) {
      nRead = fread(&pulsarcentric,sizeof(pulsarcentric),1,inputfile);
      totalbytes+=sizeof(pulsarcentric);
    } else if (strcmp(string,"nbins")) {
      nRead = fread(&nbins,sizeof(nbins),1,inputfile);
      totalbytes+=sizeof(nbins);
    } else if (strcmp(string,"nsamples")) {
      nRead = fread(&itmp,sizeof(itmp),1,inputfile);
      totalbytes+=sizeof(itmp);
    } else if (strcmp(string,"nifs")) {
      nRead = fread(&nifs,sizeof(nifs),1,inputfile);
      totalbytes+=sizeof(nifs);
    } else if (strcmp(string,"npuls")) {
      nRead = fread(&npuls,sizeof(npuls),1,inputfile);
      totalbytes+=sizeof(npuls);
    } else if (strcmp(string,"refdm")) {
      nRead = fread(&refdm,sizeof(refdm),1,inputfile);
      totalbytes+=sizeof(refdm);
    } else if (expecting_rawdatafile) {
      strcpy(rawdatafile,string);
      expecting_rawdatafile=0;
    } else if (expecting_source_name) {
      strcpy(source_name,string);
      expecting_source_name=0;
    } else {
      sprintf(message,"read_header - unknown parameter: %s\n",string);
      fprintf(stderr,"ERROR: %s\n",message);
      exit(1);
    } 
  } 


  totalbytes+=nbytes;

  return totalbytes;
}
*/

void usage()
{
  fprintf (stdout,
	   "dsaX_fake [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -f file to read packet from [default none]\n"
	   " -i in_key [default TEST_BLOCK_KEY]\n"
	   " -o out_key [default REORDER_BLOCK_KEY2]\n"
	   " -n will not read header\n"
	   " -b number of blocks to stop after\n"
	   " -h print usage\n");
}

// MAIN

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_fake", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t in_key = 0x0000dada;
  key_t out_key = 0x0000caca;
  
  // command line arguments
  int core = -1;
  int useZ = 1;
  char fnam[100];
  int arg = 0;
  int rhead = 1;
  int nblocks = -1;
  
  while ((arg=getopt(argc,argv,"c:f:i:o:nb:dh")) != -1)
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
	      useZ = 0;
	      strcpy(fnam,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'b':
	  if (optarg)
	    {
	      nblocks = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-b flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'n':
	  rhead=0;
	  syslog (LOG_INFO, "Will not read header");
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
  
  uint64_t header_size = 0;

  // deal with headers
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "could not read next header");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  
  // record STATE info
  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %llu %llu\n",block_size,block_out);
  uint64_t  bytes_read = 0;
  uint64_t npackets = 1;
  char * block, * output_buffer;
  char * packet;
  packet = (char *)malloc(sizeof(char)*block_size);
  output_buffer = (char *)malloc(sizeof(char)*block_out);
  memset(output_buffer,0,block_out);
  uint64_t written, block_id;

  // fill output buffer if file exists
  FILE *fin;
  if (!useZ) {

    if (!(fin=fopen(fnam,"rb"))) {
      syslog(LOG_ERR, "cannot open file - will write zeros");
    }
    else {
		
      if (rhead) read_header(fin);
//		fread(packet,block_out,1,fin);
//		fclose(fin);

//		syslog(LOG_INFO,"Read packet, npackets %llu",npackets);
      
//      for (int i=0;i<npackets;i++)
//		memcpy(output_buffer,packet,block_out);

//		syslog(LOG_INFO, "Using input packet");
      
    }

    
  }

  // set up

  int observation_complete=0;
  int blocks = 0, started = 0;
  
  syslog(LOG_INFO, "starting observation");
  
  /*if (!(feof(fin)) {
    fread()
	}
	else {
		close and reopen file
	}
*/

  while (!observation_complete) {
    if (!(feof(fin))) {
      fread(packet,block_out,1,fin);
    }
    else{
      fclose(fin);
      fin=fopen(fnam,"rb");
      if (rhead) read_header(fin);
      fread(packet,block_out,1,fin);
    }

    // open block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    if (started==0) {
      syslog(LOG_INFO,"now in RUN state");
      started=1;
    }

    // DO STUFF
    // no need to do anything here - output_buffer is ready to go

	// fread goes here
	// count blocks, increment, stop loop and reopen file (or rewind)

    // write to output
    written = ipcio_write (hdu_out->data_block, packet, block_out);
    if (written < block_out)
      {
		syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
		dsaX_dbgpu_cleanup (hdu_in, hdu_out);
		return EXIT_FAILURE;
      }

    if (DEBUG) {
      syslog(LOG_DEBUG, "written block %d",blocks);      
    }
    blocks++;

    if (blocks==nblocks)
      observation_complete = 1;

    ipcio_close_block_read (hdu_in->data_block, bytes_read);

  }

  fclose(fin);
  free(packet);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}
