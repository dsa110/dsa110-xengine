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

void usage()
{
  fprintf (stdout, "dsaX_splice [16 files]\n");
}

int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_splice", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());

  // set up input array
  // 16 corrs, 3840 times, 256 beams, 48 chans
  char * bigarr = (char *)malloc(sizeof(char)*16*3840*256*48);
  char foutnam[200];

  // read into input array
  FILE *fin;
  for (int i=1;i<17;i++) {
    fin=fopen(argv[i],"rb");
    fread(bigarr+(i-1)*3840*256*48,3840*256*48,1,fin);
    fclose(fin);
  }

  // reorder bigarr
  char * tarr = (char *)malloc(sizeof(char)*16*3840*256*48);
  int oidx, iidx;
  // order is beam, time, freq
  for (int i=0;i<16;i++) {
    for (int j=0;j<3840;j++) {
      for (int k=0;k<256;k++) {

	iidx = i*3840*256*48 + j*256*48 + k*48;
	oidx = k*3840*768 + j*768 + i*48;
	memcpy(tarr + oidx, bigarr + iidx, 48);

      }
    }
  }
  free(bigarr);

  // loop over beams and write out all filterbanks
  for (int i=0;i<256;i++) {
    
    sprintf(foutnam,"/home/ubuntu/data/fb_%d.fil",i);    
    
    if (!(output = fopen(foutnam,"wb"))) {
      printf("Couldn't open output file\n");
      return 0;
    }
    
    send_string("HEADER_START");
    send_string("source_name");
    sprintf(srcnam,"fb_%d",i);
    send_string(srcnam);
    send_int("machine_id",1);
    send_int("telescope_id",82);
    send_int("data_type",1); // filterbank data
    send_double("fch1",1498.75); // THIS IS CHANNEL 0 :)
    send_double("foff",-0.244140625);
    send_int("nchans",768);
    send_int("nbits",8);
    send_double("tstart",55000.0);
    send_double("tsamp",8.192e-6*8.*16.);
    send_int("nifs",1);
    send_string("HEADER_END");

    fwrite(tarr + i*2949120,2949120,1,output);
    fclose(output);

  }

  // write out full filterbank
  sprintf(foutnam,"/home/ubuntu/data/fb_all.fil");    
  
  if (!(output = fopen(foutnam,"wb"))) {
    printf("Couldn't open output file\n");
    return 0;
  }
    
  send_string("HEADER_START");
  send_string("source_name");
  sprintf(srcnam,"fb_all");
  send_string(srcnam);
  send_int("machine_id",1);
  send_int("telescope_id",82);
  send_int("data_type",1); // filterbank data
  send_double("fch1",1498.75); // THIS IS CHANNEL 0 :)
  send_double("foff",-0.244140625);
  send_int("nchans",768);
  send_int("nbits",8);
  send_double("tstart",55000.0);
  send_double("tsamp",8.192e-6*8.*16.);
  send_int("nifs",1);
  send_string("HEADER_END");
  
  fwrite(tarr,16*3840*256*48,1,output);
  fclose(output);

  
  free(tarr);
  
}
