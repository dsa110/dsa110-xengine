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

int main(int argc, char * argv[]) {

  // memory
  uint64_t bsize = 4026531840, bls = 188743680;
  unsigned char * allbeams = (unsigned char *)malloc(sizeof(unsigned char)*bsize);  
  memset(allbeams,0,bsize);
  unsigned char * data = (unsigned char *)malloc(sizeof(unsigned char)*bls);  
  FILE *fin;
  
  // load in data if present
  for (int i=0;i<16;i++) {

    if (strcmp(argv[i+1],"none")!=0) {
    
      fin=fopen(argv[i+1],"rb");
      fread(data,sizeof(unsigned char),bls,fin);
      fclose(fin);      
      
      for (int ibeam=0;ibeam<256;ibeam++) {
	for (int itime=0;itime<30*512;itime++) {
	  for (int ich=0;ich<48;ich++) {
	    allbeams[ibeam*30*512*1024 + itime*1024 + i*48 + ich + 128] = data[itime*256*48 + ibeam*48 + ich];
	  }
	}
      }
    }
    
  }

  // make files

  char cmd[300], foutnam[400];
  sprintf(cmd,"mkdir -p %s_%s",argv[17],argv[18]);
  system(cmd);

  for (int i=0;i<256;i++) {
	  
    sprintf(foutnam,"%s_%s/%s_%d.fil",argv[17],argv[18],argv[18],i);
    output = fopen(foutnam,"wb");
    
    send_string("HEADER_START");
    send_string("source_name");
    send_string(argv[18]);
    send_int("machine_id",1);
    send_int("telescope_id",82);
    send_int("data_type",1); // filterbank data
    send_double("fch1",1530.0); // THIS IS CHANNEL 0 :)
    send_double("foff",-0.244140625);
    send_int("nchans",1024);
    send_int("nbits",8);
    send_double("tstart",55000.0);
    send_double("tsamp",8.192e-6*8.*4.);
    send_int("nifs",1);
    send_string("HEADER_END");
	  
    fwrite(allbeams + i*30*512*1024,sizeof(unsigned char),30*512*1024,output);
	  
    fclose(output);
	  
  }

  
  free(allbeams);
  free(data);

}
