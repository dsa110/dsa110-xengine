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
//#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include <math.h>
#include <syslog.h>

// for random numbers generation
//#include <cmath>
//#include <cstdlib>

//#include "ascii_header.h"

#define PORT 4444 

#define NTIMES 4096	// # of time samples (assuming 1ms sampling period)
#define NCHAN 1024	// # of channels on BF node side
#define NBEAMS 64	// # of beams on BF side
#define M NTIMES
#define N 32
#define HDR_SIZE 4096
#define BUF_SIZE NTIMES*NCHAN*NBEAMS // size of TCP packet

double skarray[NBEAMS*NCHAN+1];	// array with SK values -- size NCHANS * NBEAMS
double avgspec[NBEAMS*NCHAN+1];	// spectrum over all beams to estimate median filter
double baselinecorrec[NBEAMS*NCHAN+1];	// spectrum over all beams to estimate median filter
	
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


int main(int argc, char**argv)
{

  openlog ("flagger", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  
// CONNECT AND READ FROM BUFFER

	dada_hdu_t* hdu_in = 0;	// header and data unit
	//multilog_t* log = 0;	// logger
	key_t in_key = 0x0000dada;	// key to connect to the buffer
	uint64_t blocksize = NTIMES*NCHAN*NBEAMS;	// size of buffer
	//log = multilog_open ("dsaX_flagger", 0);
	//multilog_add (log, stderr);	// log stderr
	hdu_in  = dada_hdu_create (log);
	dada_hdu_set_key (hdu_in, in_key);
	if (dada_hdu_connect (hdu_in) < 0) {
		printf ("could not connect to input buffer\n");
		return EXIT_FAILURE;
	}
	if (dada_hdu_lock_read (hdu_in) < 0) {
		printf ("could not lock to input buffer\n");
		return EXIT_FAILURE;
	}
	
	uint64_t header_size = 0;
	// read the header from the input HDU
	char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
	
	// mark the input header as cleared
	if (ipcbuf_mark_cleared (hdu_in->header_block) < 0){
		printf ("could not mark header as cleared\n");
		return EXIT_FAILURE;
	}

	uint64_t block_id, bytes_read = 0;
//	char * cpbuf = (char *)malloc(sizeof(char)*blocksize);
	char *in_data;
	
	
	
// OUTPUT BUFFER
	dada_hdu_t* hdu_out = 0;
	//multilog_t* log2 = 0;
	key_t out_key2 = 0x0000caca;
	//	log2 = multilog_open ("flagged_data", 0);
	//multilog_add (log2, stderr);
	//multilog (log2, LOG_INFO, "flagged_data: creating hdu\n");  
	hdu_out  = dada_hdu_create (log2);
	dada_hdu_set_key (hdu_out, out_key2);
	if (dada_hdu_connect (hdu_out) < 0) {
		printf ("flagged_data: could not connect to dada buffer\n");
		return EXIT_FAILURE;
	}
	if (dada_hdu_lock_write (hdu_out) < 0) {
		printf ("flagged_data: could not lock to dada buffer\n");
		return EXIT_FAILURE;
	}
	
	/*read fake header for now*/
	char head_dada[4096];
	FILE *f = fopen("/home/dsa/dsa110-xengine/src/correlator_header_dsaX.txt", "rb");
	fread(head_dada, sizeof(char), 4096, f);
	fclose(f);
			//// OUTPUT BUFFER
		char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
		header_size = HDR_SIZE;
		if (!header_out)
		{
		  //multilog(log2, LOG_ERR, "could not get next header block [output]\n");
			return EXIT_FAILURE;
		}
		memcpy (header_out, head_dada, header_size);
		if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
		{
		  //multilog (log2, LOG_ERR, "could not mark header block filled [output]\n");
			return EXIT_FAILURE;
		}
		uint64_t written=0;

	////////////////
	
	
	
	double S1 = 0;
	double S2 = 0;
	double nThreshUp = 0.5;	// Threshold to apply to SK (empirical estimation)
	
	int nFiltSize = 21;
	
	
	
	while (1) {	// put rest of the code inside while loop

		// read a DADA block
		in_data = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
//		memcpy(cpbuf, in_data, blocksize);

		// close block after reading
		ipcio_close_block_read (hdu_in->data_block, bytes_read);
		syslog(LOG_INFO,"data read");		
		
		// compute SK and averaged spectrum
		S1 = 0;
		S2 = 0;
		double sampval = 0;
		
		for (int i = 0; i < NBEAMS; i++){
			for (int k = 0; k < NCHAN; k++){
				for (int j = 0; j < NTIMES; j++){
					sampval = (double)in_data[i*(int)(NCHAN*NTIMES)+j*(int)NCHAN+k];
					avgspec[i*(int)(NCHAN) + k] += sampval / NTIMES;
					S1 += sampval;
					S2 += sampval * sampval;
					skarray[i*(int)(NCHAN) + k] = (double)((M*N+1) / (M-1) * ( (M*S2)/(S1*S1) - 1 ));
				}
				S1 = 0;
				S2 = 0;
			}
		}
		syslog (LOG_INFO,"has computed SK.");
		syslog(LOG_INFO,"example SK value : %g", (double)skarray[10]);
		
		// compute baseline correction
/*		for (int i = 0; i < NBEAMS*NCHAN-nFiltSize; i++)
			baselinecorrec[i] = medval(&avgspec[i],nFiltSize);
*/		
		
		// compare SK values to threshold and
		// replace thresholded channels with noise or 0
		
		int cnt = 0;
		for (int i = 0; i < NBEAMS; i++){
			for (int k = 0; k < NCHAN; k++){
				if (skarray[i*(int)(NCHAN) + k] > nThreshUp){
				  cnt++;
					for (int j = 0; j < NTIMES; j++){
						in_data[i*(int)(NCHAN*NTIMES)+j*(int)NCHAN+k] = (unsigned char)(20 * rand() / ( (double)RAND_MAX ) + 10.);
				//		in_data[i*(int)(NCHAN*NTIMES)+j*(int)NCHAN+k] = 0;
					}
				}
			}
		}
		syslog (LOG_INFO,"has replaced values. %d channels flagged",cnt);
		
		// apply baseline correction
		/*
		for (int i = 0; i < NBEAMS; i++){
			for (int k = 0; k < NCHAN; k++){
				for (int j = 0; j < NTIMES; j++){
					in_data[i*(int)(NCHAN*NTIMES)+j*(int)NCHAN+k] = (unsigned char)(in_data[i*(int)(NCHAN*NTIMES)+j*(int)NCHAN+k] / (unsigned char)baselinecorrec[i*(int)NCHAN+k]);
				}
			}
		}
		
		printf ("baseline correction applied.\n");
		*/
		
		

		written = ipcio_write (hdu_out->data_block, in_data, BUF_SIZE);
		if (written < BUF_SIZE)
		{
		  //multilog(log2, LOG_INFO, "main: failed to write all data to datablock [output]\n");
			return EXIT_FAILURE;
		}
//		if (dada_hdu_unlock_write (hdu_out) < 0)
//		{
//			multilog(log2, LOG_ERR, "could not unlock write on hdu_out\n");
//		}
//		dada_hdu_destroy (hdu_out);

		syslog (LOG_INFO,"write flagged data done.");
		
	
	}

	return 0;    
}  
