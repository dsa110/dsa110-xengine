/* dsaX_capture_pcap.c: Code to capture packets using pf_ring aware pcap and write to a dada buffer.

control and stats threads: standard threads
recv thread: simply runs pcap_loop, passing packets to callback function
packet_callback: places packets directly into dada buffer, or temp buffer. gets new buffer if needed

everything is in the dsaX_t structure


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
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_capture_pcap.h"
#include "dsaX_def.h"
#include "pcap.h"

/* global variables */
int quit_threads = 0;
char STATE[20];
uint64_t UTC_START = 10000;
uint64_t UTC_STOP = 40000000000;
int MONITOR = 0;
char iP[100];
int DEBUG = 0;
int HISTOGRAM[16];
int cores[2] = {17,19};
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int canWrite = 0;
volatile  unsigned capture_started = 0;
volatile char * wblock;
volatile uint64_t last_seq;
volatile uint64_t writeBlock = 0;
const int nth = 1;
const int nwth = 1;
const int TEMP_MAXY = 1000;

void dsaX_dbgpu_cleanup (dada_hdu_t * out);
int dada_bind_thread_to_core (int core);
void usage();

void dsaX_dbgpu_cleanup (dada_hdu_t * out)
{

  if (dada_hdu_unlock_write (out) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_out");
    }
  dada_hdu_destroy (out);  
  
}

void usage()
{
  fprintf (stdout,
	   "dsaX_capture [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -i IP to listen on for control commands [no default]\n"	
	   " -f filename of template dada header [no default]\n"
	   " -o out_key [default CAPTURE_BLOCK_KEY]\n"
	   " -d send debug messages to syslog\n"
	   " -h print usage\n");
}

/* 
 *  open a data block buffer ready for direct access
 */
int dsaX_udpdb_open_buffer (dsaX_t * ctx);
int dsaX_udpdb_open_buffer (dsaX_t * ctx)
{

  if (DEBUG) syslog (LOG_DEBUG, "dsaX_udpdb_open_buffer()");

  if (ctx->block_open)
  {
    syslog (LOG_ERR, "open_buffer: buffer already opened");
    return -1;
  }

  if (DEBUG) syslog (LOG_DEBUG, "open_buffer: ipcio_open_block_write");

  uint64_t block_id = 0;

  wblock = ipcio_open_block_write (ctx->hdu->data_block, &block_id);
  if (!wblock)
  { 
    syslog (LOG_ERR, "open_buffer: ipcio_open_block_write failed");
    return -1;
  }

  ctx->block_open = 1;

  return 0;
}

/*
 *  close a data buffer, assuming a full block has been written
 */
int dsaX_udpdb_close_buffer (dsaX_t * ctx, uint64_t bytes_written, unsigned eod);
int dsaX_udpdb_close_buffer (dsaX_t * ctx, uint64_t bytes_written, unsigned eod)
{

  if (DEBUG) syslog (LOG_DEBUG, "dsaX_udpdb_close_buffer(%"PRIu64", %d)", bytes_written, eod);

  if (!ctx->block_open)
  { 
    syslog (LOG_ERR, "close_buffer: buffer already closed");
    return -1;
  }

  // log any buffers that are not full, except for the 1 byte "EOD" buffer
  if ((bytes_written != 1) && (bytes_written != ctx->hdu_bufsz))
    syslog ((eod ? LOG_INFO : LOG_WARNING), "close_buffer: "
              "bytes_written[%"PRIu64"] != hdu_bufsz[%"PRIu64"]", 
              bytes_written, ctx->hdu_bufsz);

  if (eod)
  {
    if (ipcio_update_block_write (ctx->hdu->data_block, bytes_written) < 0)
    {
      syslog (LOG_ERR, "close_buffer: ipcio_update_block_write failed");
      return -1;
    }
  }
  else 
  {
    if (ipcio_close_block_write (ctx->hdu->data_block, bytes_written) < 0)
    {
      syslog (LOG_ERR, "close_buffer: ipcio_close_block_write failed");
      return -1;
    }
  }

  wblock = 0;
  ctx->block_open = 0;

  return 0;
}

/* 
 *  move to the next ring buffer element. return pointer to base address of new buffer
 */
int dsaX_udpdb_new_buffer (dsaX_t * ctx);
int dsaX_udpdb_new_buffer (dsaX_t * ctx)
{

  if (DEBUG) syslog (LOG_DEBUG, "dsaX_udpdb_new_buffer()");

  if (dsaX_udpdb_close_buffer (ctx, ctx->hdu_bufsz, 0) < 0)
  {
    syslog (LOG_ERR, "new_buffer: dsaX_udpdb_close_buffer failed");
    return -1;
  }

  if (dsaX_udpdb_open_buffer (ctx) < 0) 
  {
    syslog (LOG_ERR, "new_buffer: dsaX_udpdb_open_buffer failed");
    return -1;
  }

  return 0;

}

// increment counters when block is full
void dsaX_udpdb_increment (dsaX_t * ctx);
void dsaX_udpdb_increment (dsaX_t * ctx)
{

  // increment buffer byte markers
  ctx->block_start_byte = ctx->block_end_byte + UDP_DATA;
  ctx->block_end_byte = ctx->block_start_byte + ( ctx->packets_per_buffer - 1) * UDP_DATA;
  ctx->block_count = 0;

}



/* --------- THREADS -------- */

// STATS THREAD

/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) {
  
  dsaX_stats_t * ctx = (dsaX_stats_t *) arg;
  uint64_t b_rcv_total = 0;
  uint64_t b_rcv_1sec = 0;
  uint64_t b_rcv_curr = 0;

  uint64_t b_drp_total = 0;
  uint64_t b_drp_1sec = 0;
  uint64_t b_drp_curr = 0;

  uint64_t s_rcv_total = 0;
  uint64_t s_rcv_1sec = 0;
  uint64_t s_rcv_curr = 0;

  uint64_t ooo_pkts = 0;
  float gb_rcv_ps = 0;
  float mb_rcv_ps = 0;
  float mb_drp_ps = 0;

  syslog(LOG_INFO,"starting stats thread...");
  sleep(2);
  syslog(LOG_INFO,"started stats thread...");
  
  while (!quit_threads)
  {

    /* get a snapshot of the data as quickly as possible */
    b_rcv_curr = ctx->bytes->received;
    b_drp_curr = ctx->bytes->dropped;
    
    /* calc the values for the last second */
    b_rcv_1sec = b_rcv_curr - b_rcv_total;
    b_drp_1sec = b_drp_curr - b_drp_total;

    /* update the totals */
    b_rcv_total = b_rcv_curr;
    b_drp_total = b_drp_curr;

    mb_rcv_ps = (double) b_rcv_1sec / 1000000;
    mb_drp_ps = (double) b_drp_1sec / 1000000;
    gb_rcv_ps = b_rcv_1sec * 8;
    gb_rcv_ps /= 1000000000;    

    /* determine how much memory is free in the receivers */
    syslog (LOG_NOTICE,"CAPSTATS %6.3f [Gb/s], D %4.1f [MB/s], D %"PRIu64" pkts, %"PRIu64" skipped 0", gb_rcv_ps, mb_drp_ps, ctx->packets->dropped, last_seq);

    sleep(1);
  }

}

// CONTROL THREAD

void control_thread (void * arg) {

  syslog(LOG_INFO, "control_thread: starting");

  // port on which to listen for control commands
  int port = CAPTURE_CONTROL_PORT;
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
  uint64_t tmps;
  char * token;
  
  syslog(LOG_INFO, "control_thread: created socket on port %d", port);
  
  while (!quit_threads) {
    
    fd = socket(res->ai_family,res->ai_socktype,res->ai_protocol);
    bind(fd,res->ai_addr,res->ai_addrlen);
    memset(buffer,'\0',sizeof(buffer));
    syslog(LOG_INFO, "control_thread: waiting for packet");
    ct = recvfrom(fd,buffer,1024,0,(struct sockaddr*)&src_addr,&src_addr_len);
    
    syslog(LOG_INFO, "control_thread: received buffer string %s",buffer);

    // INTERPRET BUFFER STRING
    // receive either UTC_START, UTC_STOP, MONITOR

    // interpret buffer string
    char * rest = buffer;
    char *cmd, *val;
    cmd = strtok_r(rest, "-", &rest);
    val = strtok_r(rest, "-", &rest);
    syslog(LOG_INFO, "control_thread: split into COMMAND %s, VALUE %s",cmd,val);

    if (strcmp(cmd,"UTC_START")==0)
      UTC_START = strtoull(val,&endptr,0);

    if (strcmp(cmd,"UTC_STOP")==0)
      UTC_STOP = strtoull(val,&endptr,0);    
    
    close(fd);
    
  }

  free (buffer);

  syslog(LOG_INFO, "control_thread: exiting");

  /* return 0 */
  int thread_result = 0;
  pthread_exit((void *) &thread_result);

}

/*
This is important - packet callback function to place packets in buffer
called upon single packet being received
*/
void packet_callback(u_char *args, const struct pcap_pkthdr* header, const u_char* packet) {

  dsaX_t * udpdb = (dsaX_t *) args;

  // make sure packet has right length and get payload
  if (header->len != UDP_PAYLOAD + 42) {
    syslog(LOG_INFO,"received packet with length %d, total available %d",header->len,header->caplen);
    return;
  }
  char *buf = (char *)(packet + 42);
  
  // process packet header
  uint64_t seq_no=0, ant_id=0;
  seq_no |=  (((uint64_t)(buf[4]) & 224) >> 5) & 7;
  seq_no |=  (((uint64_t)(buf[3])) << 3) & 2040;
  seq_no |=  (((uint64_t)(buf[2])) << 11) & 522240;
  seq_no |=  (((uint64_t)(buf[1])) << 19) & 133693440;
  seq_no |=  (((uint64_t)(buf[0])) << 27) & 34225520640;
  ant_id |= (unsigned char) (buf[6]) << 8;
  ant_id |= (unsigned char) (buf[7]);	  
  uint64_t act_seq_no = seq_no*NCHANG*NSNAPS/2 + ant_id*NCHANG/3; // actual seq no
  uint64_t block_seq_no = UTC_START*NCHANG*NSNAPS/2; // seq no corresponding to ant 0 and start of block
  last_seq = seq_no;
    
  // check for starting condition
  if (canWrite==0) {
    if (seq_no >= UTC_START-50 && UTC_START != 10000) {
      canWrite=1;	      
    }
  }
  if (canWrite == 0) return;

  // deal with start of capture
  if (!(capture_started))
    {
      udpdb->block_start_byte = block_seq_no * UDP_DATA;
      udpdb->block_end_byte   = (udpdb->block_start_byte + udpdb->hdu_bufsz) - UDP_DATA;
      capture_started = 1;      
      syslog (LOG_INFO, "receive_obs: START [%"PRIu64" - %"PRIu64"]", udpdb->block_start_byte, udpdb->block_end_byte);
    }

  // if capture has started, do good stuff
  uint64_t byte_offset, seq_byte;
  if (capture_started) {

    seq_byte = (act_seq_no * UDP_DATA);

    // packet belongs in this block
    if ((seq_byte <= udpdb->block_end_byte) && (seq_byte >= udpdb->block_start_byte))
      {
	byte_offset = seq_byte - (udpdb->block_start_byte);
	memcpy(udpdb->tblock + udpdb->tblock_idx*UDP_DATA + byte_offset, buf + UDP_HEADER, UDP_DATA);	
	//memcpy(wblock + byte_offset, buf + UDP_HEADER, UDP_DATA);
	udpdb->block_count++;
      }
    // packet belongs in subsequent block
    else if (seq_byte > udpdb->block_end_byte)
      {
	if (udpdb->temp_idx < TEMP_MAXY)
	  {
	    // save packet to temp buffer
	    memcpy (udpdb->temp_buffers + udpdb->temp_idx*UDP_DATA, buf + UDP_HEADER, UDP_DATA);
	    udpdb->temp_seq_byte[udpdb->temp_idx] = seq_byte;
	    udpdb->temp_idx++;
	  }
      }
  }

  // end of block
  if ((udpdb->block_count >= udpdb->packets_per_buffer) || (udpdb->temp_idx >= TEMP_MAXY))
    {
      syslog (LOG_INFO, "BLOCK COMPLETE seq_no=%"PRIu64", "
	      "ant_id=%"PRIu16", block_count=%"PRIu64", "
	      "temp_idx=%d", seq_no, ant_id,
	      udpdb->block_count, udpdb->temp_idx);

      // set writeBlock
      if (udpdb->tblock_idx==0) {
	writeBlock = 1;
	udpdb->tblock_idx = NPACKETS_PER_BLOCK*NSNAPS;
      }
      else if (udpdb->tblock_idx==NPACKETS_PER_BLOCK*NSNAPS) {
	writeBlock = 2;
	udpdb->tblock_idx = 0;
      }
      /*
      // get new block
      if (dsaX_udpdb_new_buffer (udpdb) < 0)
	{
	  syslog(LOG_ERR, "receive_obs: dsaX_udpdb_new_buffer failed");
	  return EXIT_FAILURE;
	}
      */
      // deal with counters
      uint64_t dropped = udpdb->packets_per_buffer - (udpdb->block_count);
      udpdb->packets->received += (udpdb->block_count);
      udpdb->bytes->received += (udpdb->block_count) * UDP_DATA;
      if (dropped)
	{
	  udpdb->packets->dropped += dropped;
	  udpdb->bytes->dropped += (dropped * UDP_DATA);
	}
      dsaX_udpdb_increment(udpdb);

      // write temp queue
      for (int i=0; i < udpdb->temp_idx; i++) {
	seq_byte = udpdb->temp_seq_byte[i];
	byte_offset = seq_byte - udpdb->block_start_byte;
	if (byte_offset < udpdb->hdu_bufsz && byte_offset >= 0) {
	  //memcpy(wblock + byte_offset, udpdb->temp_buffers + i*UDP_DATA, UDP_DATA);
	  memcpy(udpdb->tblock + udpdb->tblock_idx*UDP_DATA + byte_offset, udpdb->temp_buffers + i*UDP_DATA, UDP_DATA);
	  udpdb->block_count++;
	}
      }
      udpdb->temp_idx = 0;

    }	  
 
}

// Thread to do writing

void write_thread(void * arg) {

  dsaX_t * udpdb = (dsaX_t *) arg;
  int thread_id = 2;

  // set affinity
  const pthread_t pid = pthread_self();
  const int core_id = cores[1];
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  const int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (set_result != 0)
    syslog(LOG_ERR,"thread %d: setaffinity_np fail",core_id);
  const int get_affinity = pthread_getaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (get_affinity != 0) 
    syslog(LOG_ERR,"thread %d: getaffinity_np fail",core_id);
  if (CPU_ISSET(core_id, &cpuset))
    syslog(LOG_INFO,"thread %d: successfully set thread",core_id);

  int a;
  while (!quit_threads) {

    // busywait
    while (writeBlock==0)
      a=1;

    // write block
    memcpy(wblock, udpdb->tblock + (writeBlock-1)*UDP_DATA*NSNAPS*NPACKETS_PER_BLOCK, UDP_DATA*NSNAPS*NPACKETS_PER_BLOCK);

    // get new block
    if (dsaX_udpdb_new_buffer (udpdb) < 0)
      {
	syslog(LOG_ERR, "receive_obs: dsaX_udpdb_new_buffer failed");
	return EXIT_FAILURE;
      }

    writeBlock = 0;
    
  }
}

/*
Thread to run pcap, passing to callback function
*/

void pcap_thread(void * arg) {

  dsaX_t * udpdb = (dsaX_t *) arg;
  int thread_id = 1;//udpdb->thread_id;
    
  // set affinity
  const pthread_t pid = pthread_self();
  const int core_id = cores[0];
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  const int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (set_result != 0)
    syslog(LOG_ERR,"thread %d: setaffinity_np fail",core_id);
  const int get_affinity = pthread_getaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
  if (get_affinity != 0) 
    syslog(LOG_ERR,"thread %d: getaffinity_np fail",core_id);
  if (CPU_ISSET(core_id, &cpuset))
    syslog(LOG_INFO,"thread %d: successfully set thread",core_id);

  // set up pcap from port CAPTURE_PORT
  char dev[] = "eth0";
  pcap_t *handle;
  char error_buffer[PCAP_ERRBUF_SIZE];
  struct bpf_program filter;
  char filter_exp[] = "port 4011";
  bpf_u_int32 subnet_mask, ip;

  if (pcap_lookupnet(dev, &ip, &subnet_mask, error_buffer) == -1) {
    syslog(LOG_ERR,"Could not get information for device: %s", dev);
    ip = 0;
    subnet_mask = 0;
  }
  handle = pcap_open_live(dev, 4659, 0, 1000, error_buffer);
  if (handle == NULL) {
    syslog(LOG_ERR,"Could not open %s - %s", dev, error_buffer);
    return 2;
  }
  
  if (pcap_compile(handle, &filter, filter_exp, 1, ip) == -1) {
    syslog(LOG_ERR,"Bad filter - %s", pcap_geterr(handle));
    return 2;
  }
  if (pcap_setfilter(handle, &filter) == -1) {
    syslog(LOG_ERR,"Error setting filter - %s\n", pcap_geterr(handle));
    return 2;
  }

  /*  if((pcap_set_buffer_size(handle, 2*1024*1024))!=0)
    {
      syslog(LOG_ERR, "Could not set buffer size");
      return 2;
      }*/

  
  syslog(LOG_INFO,"thread %d: successfully set up pcap",thread_id);

  // start up RX!
  while (!quit_threads)
    pcap_loop(handle, 0, packet_callback, (u_char*)udpdb);

  // finish
  pcap_close(handle);
  
}


	    
// MAIN of program
	
int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_capture_pcap", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit for writing */
  dada_hdu_t* hdu_out = 0;
  
  // input data block HDU key
  key_t out_key = CAPTURE_BLOCK_KEY;

  // command line arguments
  int core = -1;
  int arg=0;
  char dada_fnam[200]; // filename for dada header
  
  while ((arg=getopt(argc,argv,"c:i:f:o:dh")) != -1)
    {
      switch (arg)
	{
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
	case 'i':
	  if (optarg)
	    {	      
	      strcpy(iP,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-i flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
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
	case 'f':
	  if (optarg)
	    {	      
	      strcpy(dada_fnam,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }	 
	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // START THREADS
  
  // start control thread
  int rval = 0;
  pthread_t control_thread_id;
  dsaX_t temp_str;
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &temp_str);
  if (rval != 0) {
    syslog(LOG_ERR, "Error creating control_thread: %s", strerror(rval));
    return -1;
  }
  syslog(LOG_NOTICE, "Created control thread, listening on %s:%d",iP,CAPTURE_CONTROL_PORT);
  
  // Bind to cpu core
  if (core >= 0)
    {
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
      syslog(LOG_NOTICE,"bound to core %d", core);
    }
  
  
  // OPEN CONNECTION TO DADA DB FOR WRITING

  if (DEBUG) syslog(LOG_DEBUG,"Creating HDU");
  
  hdu_out  = dada_hdu_create ();
  if (DEBUG) syslog(DEBUG,"Created hdu");
  dada_hdu_set_key (hdu_out, CAPTURE_BLOCK_KEY);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog(LOG_ERR,"could not connect to output dada buffer");
    return EXIT_FAILURE;
  }
  if (DEBUG) syslog(LOG_DEBUG,"Connected HDU");
  if (dada_hdu_lock_write(hdu_out) < 0) {
    dsaX_dbgpu_cleanup (hdu_out);
    syslog(LOG_ERR,"could not lock to output dada buffer");
    return EXIT_FAILURE;
  }

  syslog(LOG_INFO,"opened connection to output DB");

  // DEAL WITH DADA HEADER
  char *hout;
  hout = (char *)malloc(sizeof(char)*4096);
  if (DEBUG) syslog(DEBUG,"read header2");

  if (fileread (dada_fnam, hout, 4096) < 0)
    {
      free (hout);
      syslog (LOG_ERR, "could not read ASCII header from %s", dada_fnam);
      return (EXIT_FAILURE);
    }

  
  if (DEBUG) syslog(DEBUG,"read header3");
  
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_out);
      return EXIT_FAILURE;
    }
  
  // copy the in header to the out header
  memcpy (header_out, hout, 4096);

  // mark the output header buffer as filled
  if (ipcbuf_mark_filled (hdu_out->header_block, 4096) < 0)
    {
      syslog(LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_out);
      return EXIT_FAILURE;
    }

  // record STATE info
  sprintf(STATE,"LISTEN");
  syslog(LOG_INFO,"marked output header block as filled - now in LISTEN state");


  /* time to start up receiver. 
  */

  // make recv, write, and stats structs  
  dsaX_t udpdb[nth];
  dsaX_stats_t stats;

  // shared variables and memory
  uint64_t bufsz = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);  
  stats_t * packets = init_stats_t();
  stats_t * bytes = init_stats_t();
  reset_stats_t(packets);
  reset_stats_t(bytes);
  char * tblock = (char *)malloc(sizeof(char)*2*(ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block)));
  char * temp_buffers = (char *)malloc(sizeof(char)*TEMP_MAXY*UDP_DATA);
  char * temp_seq_byte = (uint64_t *)malloc(sizeof(uint64_t)*TEMP_MAXY);
  
  // initialise stats struct
  stats.packets = packets;
  stats.bytes = bytes;

  for (int i=0;i<nth;i++) {

    udpdb[i].hdu = hdu_out;
    udpdb[i].hdu_bufsz = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
    udpdb[i].block_open = 0;
    udpdb[i].block_count = 0;
    udpdb[i].tblock = tblock;
    udpdb[i].tblock_idx = 0;
    udpdb[i].temp_buffers = temp_buffers;
    udpdb[i].temp_seq_byte = temp_seq_byte;
    udpdb[i].temp_idx = 0;
    udpdb[i].thread_id = 1;
    udpdb[i].verbose = 0;
    udpdb[i].packets_per_buffer = udpdb[i].hdu_bufsz / UDP_DATA;
    udpdb[i].packets = packets;
    udpdb[i].bytes = bytes;

  }    
  dsaX_udpdb_open_buffer (&udpdb[0]);

  /* start threads */
    
  // start the stats thread
  pthread_t stats_thread_id;
  rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &stats);
  if (rval != 0) {
    syslog(LOG_INFO, "Error creating stats_thread: %s", strerror(rval));
    return -1;
  }
  syslog(LOG_NOTICE, "started stats_thread()");

  // start the receive threads
  pthread_t recv_thread_id[nth];  
  rval = 0;
  for (int i=0;i<nth;i++) {
    rval = pthread_create (&recv_thread_id[i], 0, (void *) pcap_thread, (void *) (&udpdb[i]));
    if (rval != 0) {
      syslog(LOG_ERR, "Error creating recv_thread %d: %s", i,strerror(rval));
      return -1;
    }
  }
  syslog(LOG_NOTICE, "Created recv threads");

  // start the write threads
  pthread_t write_thread_id[nwth];  
  rval = 0;
  for (int i=0;i<nwth;i++) {
    rval = pthread_create (&write_thread_id[i], 0, (void *) write_thread, (void *) (&udpdb[i]));
    if (rval != 0) {
      syslog(LOG_ERR, "Error creating write_thread %d: %s", i,strerror(rval));
      return -1;
    }
  }
  syslog(LOG_NOTICE, "Created write threads");

  
  while (!quit_threads) {
    sleep(1);
  }
  
  // close threads
  syslog(LOG_INFO, "joining all threads");
  quit_threads = 1;
  void* result=0;
  pthread_join (control_thread_id, &result);
  pthread_join (stats_thread_id, &result);
  for (int i=0;i<nth;i++) pthread_join(recv_thread_id[i], &result);
  for (int i=0;i<nwth;i++) pthread_join(write_thread_id[i], &result);
  
  free(tblock);
  free(temp_buffers);
  free(temp_seq_byte);
  dsaX_dbgpu_cleanup (hdu_out);

}
