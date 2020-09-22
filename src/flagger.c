#include "stdio.h"  
#include "stdlib.h"  
#include "sys/types.h"  
#include "sys/socket.h"  
#include "string.h"  
#include "netinet/in.h"  
#include "netdb.h"
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include <math.h>
#include <syslog.h>
#include "dsaX_def.h"

#define NTIMES_P 4096	// # of time samples (assuming 1ms sampling period)
#define NCHAN_P 1024	// # of channels on BF node side
#define NBEAMS_P 64	// # of beams on BF side
#define M_P NTIMES_P
#define N_P 32
#define HDR_SIZE 4096
#define BUF_SIZE NTIMES_P*NCHAN_P*NBEAMS_P // size of TCP packet

// global variables
int DEBUG = 0;
double skarray[NBEAMS_P*NCHAN_P+1];	// array with SK values -- size NCHANS * NBEAMS
double avgspec[NBEAMS_P*NCHAN_P+1];	// spectrum over all beams to estimate median filter
double baselinecorrec[NBEAMS_P*NCHAN_P+1];	// spectrum over all beams to estimate median filter
	
void swap(char *p,char *q) {
   char t;
   
   t=*p; 
   *p=*q; 
   *q=t;
}

double medval(double a[],int n) { 
	int i,j;
	char tmp[n];
	for (i = 0;i < n;i++)
		tmp[i] = a[i];
	
	for(i = 0;i < n-1;i++) {
		for(j = 0;j < n-i-1;j++) {
			if(tmp[j] > tmp[j+1])
				swap(&tmp[j],&tmp[j+1]);
		}
	}
	return tmp[(n+1)/2-1];
}

void usage()
{
  fprintf (stdout,
	   "flagger [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default dada]\n"
	   " -o out_key [default caca]\n"
	   " -n use noise generation rather than zeros\n"
	   " -t SK threshold [default 5.0]\n"
	   " -b compute and apply baseline correction\n"
	   " -h print usage\n");
}


int main(int argc, char**argv)
{

  // syslog start
  openlog ("flagger", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  // read command line args

  // data block HDU keys
  key_t in_key = 0x0000dada;
  key_t out_key = 0x0000caca;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  int noise = 0;
  double skthresh = 5.0;
  int bcorr = 0;
  
  while ((arg=getopt(argc,argv,"c:t:i:o:bndh")) != -1)
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
	case 't':
	  if (optarg)
	    {
	      skthresh = atof(optarg);
	      syslog(LOG_INFO,"modified SKTHRESH to %g",skthresh);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-t flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }

	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'n':
	  noise=1;
	  syslog (LOG_INFO, "Will generate noise samples");
	  break;	  
	case 'b':
	  bcorr=1;
	  syslog (LOG_INFO, "Will calculate and apply baseline correction");
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
  
  
  // CONNECT AND READ FROM BUFFER

  dada_hdu_t* hdu_in = 0;	// header and data unit
  uint64_t blocksize = NTIMES_P*NCHAN_P*NBEAMS_P;	// size of buffer
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
  
  uint64_t header_size = 0;
  // read the header from the input HDU
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  
  // mark the input header as cleared
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0){
    syslog (LOG_ERR,"could not mark header as cleared");
    return EXIT_FAILURE;
  }
  
  uint64_t block_id, bytes_read = 0;
  unsigned char *in_data;
  char *cin_data;
	     	
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
	
  /* //read fake header for now
	char head_dada[4096];
	FILE *f = fopen("/home/dsa/dsa110-xengine/src/correlator_header_dsaX.txt", "rb");
	fread(head_dada, sizeof(char), 4096, f);
	fclose(f); */
  
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
  
  ////////////////		
	
  double S1 = 0;
  double S2 = 0;
  double sampval;
  double nThreshUp = skthresh;	// Threshold to apply to SK (empirical estimation)
	
  int nFiltSize = 21;

  // put rest of the code inside while loop
  while (1) {	
    
    // read a DADA block
    cin_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    in_data = (unsigned char *)(cin_data);
    
    // compute SK and averaged spectrum
    S1 = 0;
    S2 = 0;
    sampval = 0;
		
    for (int i = 0; i < NBEAMS_P; i++){
      for (int k = 0; k < NCHAN_P; k++){
	for (int j = 0; j < NTIMES_P; j++){
	  sampval = (double)in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k];
	  avgspec[i*(int)(NCHAN_P) + k] += sampval / NTIMES_P;
	  S1 += sampval;
	  S2 += sampval * sampval;
	  skarray[i*(int)(NCHAN_P) + k] = (double)((M_P*N_P+1) / (M_P-1) * ( (M_P*S2)/(S1*S1) - 1 ));
	}
	S1 = 0;
	S2 = 0;
      }
    }
    if (DEBUG) syslog (LOG_DEBUG,"has computed SK.");
    if (DEBUG) syslog(LOG_DEBUG,"example SK value : %g", (double)skarray[10]);
		
    // compute baseline correction
    if (bcorr) {
      for (int i = 0; i < NBEAMS_P*NCHAN_P-nFiltSize; i++)
	baselinecorrec[i] = medval(&avgspec[i],nFiltSize);
    }
    		
    
    // compare SK values to threshold and
    // replace thresholded channels with noise or 0
    
    int cnt = 0;
    for (int i = 0; i < NBEAMS_P; i++){
      for (int k = 0; k < NCHAN_P; k++){
	if (skarray[i*(int)(NCHAN_P) + k] > nThreshUp){
	  cnt++;
	  for (int j = 0; j < NTIMES_P; j++){
	    if (noise)
	      in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = (unsigned char)(20 * rand() / ( (double)RAND_MAX ) + 10.);
	    else
	      in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = 0;
	  }
	}
      }
    }
    syslog (LOG_INFO,"%d channels*baselines flagged",cnt);
		
    // apply baseline correction
    if (bcorr) {
      for (int i = 0; i < NBEAMS_P; i++){
	for (int k = 0; k < NCHAN_P; k++){
	  for (int j = 0; j < NTIMES_P; j++){
	    //in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = (unsigned char)(in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] / (unsigned char)baselinecorrec[i*(int)NCHAN_P+k]);
	    in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k] = (unsigned char)((double)(in_data[i*(int)(NCHAN_P*NTIMES_P)+j*(int)NCHAN_P+k]) / baselinecorrec[i*(int)NCHAN_P+k]);
	  }
	}
      }
      
      syslog (LOG_DEBUG,"baseline correction applied");
    }
		
    // close block after reading
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    if (DEBUG) syslog(LOG_DEBUG,"closed read block");		
    
    written = ipcio_write (hdu_out->data_block, (char *)(in_data), BUF_SIZE);
    if (written < BUF_SIZE)
      {
	syslog(LOG_ERR,"write error");
	return EXIT_FAILURE;
      }

    if (DEBUG) syslog (LOG_DEBUG,"write flagged data done.");
		
    
  }

  return 0;    
}  
