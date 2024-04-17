/* dsaX_capture.c: Code to capture packets over a socket and write to a dada buffer.

main: runs capture loop, and interfaces dada buffer
control_thread: deals with control commands

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
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/ether.h>


#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_capture_manythread.h"
#include "dsaX_def.h"

/* global variables */
int dPort, cPort;
int quit_threads = 0;
char STATE[20];
uint64_t UTC_START = 10000;
uint64_t UTC_STOP = 40000000000;
int MONITOR = 0;
char iP[100];
int DEBUG = 0;
int HISTOGRAM[16];
int writeBlock = 0;
const int nth = 4;
const int nwth = 2;
int cores[16] = {10,12,11,13,30,31,32,33};
int write_cores[8] = {14,15,34,35};
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int blockStatus[64];
volatile int skipBlock = 0;
volatile int skipping = 0;
volatile int lWriteBlock = 0;
volatile int write_ct = 0;
volatile uint64_t last_seq = 0;
//volatile uint64_t npackets = 0;
volatile int skipct = 0;
volatile uint64_t block_count = 0;
volatile uint64_t block_start_byte=0, block_end_byte=0;
volatile  unsigned capture_started = 0;
volatile char * wblock;

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
	   " -j IP to listen on for data packets [no default]\n"
	   " -i IP to listen on for control commands [no default]\n"
	   " -p PORT for data\n"
	   " -q PORT for control\n"
	   " -f filename of template dada header [no default]\n"
	   " -o out_key [default CAPTURE_BLOCK_KEY]\n"
	   " -d send debug messages to syslog\n"
	   " -g chgroup [default 0]\n"
	   " -h print usage\n");
}

// open a socket
dsaX_sock_t * dsaX_make_sock (udpdb_t * ctx);
dsaX_sock_t * dsaX_make_sock (udpdb_t * ctx)
{

  // prepare structure
  syslog(LOG_INFO, "dsaX_make_sock(): preparing sock structure");
  dsaX_sock_t * b = (dsaX_sock_t *) malloc(sizeof(dsaX_sock_t));
  assert(b != NULL);
  b->bufsz = sizeof(char) * (UDP_PAYLOAD+28);
  b->buf = (char *) malloc (b->bufsz);
  assert(b->buf != NULL);
  b->have_packet = 0;
  b->fd = 0;

  // connect to socket
  syslog(LOG_INFO, "dsaX_make_sock(): connecting to socket %s:%d", ctx->interface, dPort);

  // open socket

  struct ifreq ifr;
  struct sockaddr_ll sa;
  size_t if_name_len=strlen(ctx->interface);
  int ss = 0;
  
  syslog(LOG_INFO, "prepare: creating udp socket on %s:%d", ctx->interface, dPort);
  b->fd = socket( AF_PACKET, SOCK_DGRAM, htons(ETH_P_IP) );
  memcpy(ifr.ifr_name,ctx->interface,if_name_len);
  ifr.ifr_name[if_name_len]=0;
  ioctl( b->fd, SIOCGIFINDEX, &ifr );
  memset( &sa, 0, sizeof( sa ) );
  sa.sll_family=AF_PACKET;
  sa.sll_protocol = 0x0000;
  sa.sll_ifindex = ifr.ifr_ifindex;
  sa.sll_hatype = 0;
  sa.sll_pkttype = PACKET_HOST;

  // for multiple connections
  int one = 1;
  setsockopt(b->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &one, sizeof(one));

  // bind
  bind( b->fd ,(const struct sockaddr *)&sa, sizeof( sa ) );
  ss = setsockopt( b->fd, SOL_SOCKET, SO_BINDTODEVICE, ctx->interface, strlen(ctx->interface) );
  
  // set the socket size to 64 MB
  int sock_buf_size = 64*1024*1024;
  syslog(LOG_INFO, "prepare: setting buffer size to %d", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, b->fd, ctx->verbose, sock_buf_size);

  // set the socket to non-blocking
  syslog(LOG_INFO, "prepare: setting non_block");
  sock_nonblock(b->fd);

  // clear any packets buffered by the kernel
  //syslog(LOG_INFO, "prepare: clearing packets at socket");
  //size_t cleared = dada_sock_clear_buffered_packets(b->fd, UDP_PAYLOAD);

  // clear blockStatus
  for (int i=0;i<64;i++) blockStatus[i] = 0;

  return b;
}



// close a socket
void dsaX_free_sock(dsaX_sock_t* b);
void dsaX_free_sock(dsaX_sock_t* b)
{
  b->fd = 0;
  b->bufsz = 0;
  b->have_packet =0;
  if (b->buf)
    free (b->buf);
  b->buf = 0;
}

/* 
 *  open a data block buffer ready for direct access
 */
int dsaX_udpdb_open_buffer (dsaX_write_t * ctx);
int dsaX_udpdb_open_buffer (dsaX_write_t * ctx)
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
int dsaX_udpdb_close_buffer (dsaX_write_t * ctx, uint64_t bytes_written, unsigned eod);
int dsaX_udpdb_close_buffer (dsaX_write_t * ctx, uint64_t bytes_written, unsigned eod)
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
int dsaX_udpdb_new_buffer (dsaX_write_t * ctx);
int dsaX_udpdb_new_buffer (dsaX_write_t * ctx)
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
void dsaX_udpdb_increment (udpdb_t * ctx);
void dsaX_udpdb_increment (udpdb_t * ctx)
{

  // increment buffer byte markers
  writeBlock++;
  block_start_byte = block_end_byte + UDP_DATA;
  block_end_byte = block_start_byte + ( ctx->packets_per_buffer - 1) * UDP_DATA;
  block_count = 0;

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
    syslog (LOG_NOTICE,"CAPSTATS %6.3f [Gb/s], D %4.1f [MB/s], D %"PRIu64" pkts, %"PRIu64" skipped %d", gb_rcv_ps, mb_drp_ps, ctx->packets->dropped, last_seq, skipct);

    sleep(1);
  }

}

// CONTROL THREAD

void control_thread (void * arg) {

  syslog(LOG_INFO, "control_thread: starting");

  // port on which to listen for control commands
  int port = cPort;
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
 *  Thread to capture data
 */
void recv_thread(void * arg) {

  udpdb_t * udpdb = (udpdb_t *) arg;
  int thread_id = udpdb->thread_id;
    
  // set affinity
  const pthread_t pid = pthread_self();
  int core_id;
  if (dPort==4011)
    core_id = cores[thread_id];
  else
    core_id = cores[thread_id+nth];
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

  // set up socket
  dsaX_sock_t * sock = dsaX_make_sock(udpdb);

    // lookup table for ant order
  uint64_t ant_lookup[100], vv;
  for (int i=0;i<100;i++) ant_lookup[i] = 0;
  for (int i=0;i<NSNAPS/2;i++) {
    for (int j=0;j<2;j++) {
      vv = (i*2+j)*3;
      ant_lookup[vv] = (uint64_t)(i);
    }
  }

  
  // DEFINITIONS
  uint64_t tpack = 0;
  uint64_t act_seq_no = 0;
  uint64_t block_seq_no = 0;
  uint64_t seq_no = 0;
  uint64_t ant_id = 0, aid;
  unsigned char * b = (unsigned char *) sock->buf;
  size_t got = 0; // data received from a recv_from call
  int errsv; // determine the sequence number boundaries for curr and next buffers
  int64_t byte_offset = 0; // offset of current packet in bytes from start of block
  uint64_t seq_byte = 0; // offset of current packet in bytes from start of obs
  // for "saving" out of order packets near edges of blocks
  unsigned int temp_idx = 0;
  unsigned int temp_max = 500;
  char ** temp_buffers;
  uint64_t * temp_seq_byte;
  temp_buffers = (char **)malloc(sizeof(char *)*temp_max);
  for (int i=0;i<temp_max;i++) temp_buffers[i] = (char *)malloc(sizeof(char)*UDP_DATA);
  temp_seq_byte = (uint64_t *)malloc(sizeof(uint64_t)*temp_max);
  unsigned i = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000000;
  int canWrite = 0;
  int ct_snaps=0;
  int mod_WB;
  int ctAnts = 0;

  // infinite loop to receive packets

  while (!quit_threads)
    {

      sock->have_packet = 0; 

      // incredibly tight loop to try and get a packet
      while (!sock->have_packet)
	{
	 
	  // receive 1 packet into the socket buffer
	  got = recvfrom ( sock->fd, sock->buf, UDP_PAYLOAD+28, 0, NULL, NULL );

	  if (got == UDP_PAYLOAD+28) 
	    {
	      sock->have_packet = 1;
	    } 
	  else if (got == -1) 
	    {
	      errsv = errno;
	      if (errsv == EAGAIN) 
		{
		  if (capture_started)
		    timeouts++;
		  //if (timeouts > timeout_max)
		  //syslog(LOG_INFO, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);		  
		}
	      else 
		{
		  //syslog (LOG_ERR, "receive_obs: recvfrom failed %s", strerror(errsv));
		  return EXIT_FAILURE;
		}
	    } 
	  else // we received a packet of the WRONG size, ignore it
	    {
	      syslog (LOG_NOTICE, "receive_obs: received %d bytes, expected %d", got, UDP_PAYLOAD+28);
	  }
	}
      timeouts = 0;

      // we have a valid packet within the timeout
      if (sock->have_packet) 
	{

	  // decode packet header (64 bits)
	  // 35 bits seq_no (for first spectrum in packet); 13 bits ch_id (for first channel in packet); 16 bits ant ID (for first antenna in packet)
	  seq_no = 0;
	  seq_no |=  (((uint64_t)(sock->buf[4+28]) & 224) >> 5) & 7;
	  seq_no |=  (((uint64_t)(sock->buf[3+28])) << 3) & 2040;
	  seq_no |=  (((uint64_t)(sock->buf[2+28])) << 11) & 522240;
	  seq_no |=  (((uint64_t)(sock->buf[1+28])) << 19) & 133693440;
	  seq_no |=  (((uint64_t)(sock->buf[0+28])) << 27) & 34225520640;
	  ant_id = 0;
	  ant_id |= (unsigned char) (sock->buf[6+28]) << 8;
	  ant_id |= (unsigned char) (sock->buf[7+28]);
	  aid = ant_lookup[(int)(ant_id)];
	  //aid = ant_id/3;
	  
	  if (UTC_START==0) UTC_START = seq_no+30000;
	  
	  act_seq_no = seq_no*NSNAPS/4 + aid; // actual seq no
	  block_seq_no = UTC_START*NSNAPS/4; // seq no corresponding to ant 0 and start of block

	  // set shared last_seq
	  pthread_mutex_lock(&mutex);
	  last_seq = seq_no;
	  //npackets++;
	  //syslog(LOG_INFO,"last_seq %"PRIu64"",last_seq);
	  pthread_mutex_unlock(&mutex);
	  
	  // check for starting or stopping condition, using continue
	  if (canWrite==0) {
	    if (seq_no >= UTC_START-50 && UTC_START != 10000) {
	      canWrite=1;	      
	    }
	  }
	  if (canWrite == 0) continue;

	  // threadsafe start of capture
	  pthread_mutex_lock(&mutex);
	  if (!(capture_started))
	    {
	      block_start_byte = block_seq_no * UDP_DATA;
	      block_end_byte   = (block_start_byte + udpdb->hdu_bufsz) - UDP_DATA;
	      capture_started = 1;

	      syslog (LOG_INFO, "receive_obs: START [%"PRIu64" - %"PRIu64"]", block_start_byte, block_end_byte);
	    }
	  pthread_mutex_unlock(&mutex);

	  // if capture running
	  if (capture_started)
	    {
	      seq_byte = (act_seq_no * UDP_DATA);
	      tpack++;
	      
	      // packet belongs in this block
	      if ((seq_byte <= block_end_byte) && (seq_byte >= block_start_byte))
		{
		  byte_offset = seq_byte - (block_start_byte);
		  mod_WB = writeBlock % 64;
		  memcpy (udpdb->tblock + byte_offset + mod_WB*udpdb->hdu_bufsz, sock->buf + UDP_HEADER+28, UDP_DATA);		  
		  pthread_mutex_lock(&mutex);		  
		  block_count++;
		  //syslog(LOG_INFO,"block count %"PRIu64"",block_count);
		  pthread_mutex_unlock(&mutex);
		  
		}
	      // packet belongs in subsequent block
	      else if (seq_byte > block_end_byte)
		{
		      
		  if (temp_idx < temp_max)
		    {
		      // save packet to temp buffer
		      memcpy (temp_buffers[temp_idx], sock->buf + UDP_HEADER+28, UDP_DATA);
		      temp_seq_byte[temp_idx] = seq_byte;
		      temp_idx++;
		    }
		}
	      // packet is too late
	      /*else
		{
		  if (ctAnts<100) {
		    syslog (LOG_INFO, "receive_obs: TOO LATE %"PRIu64"  %"PRIu64"", seq_no, ant_id);
		    ctAnts++;
		  }
		  }*/
	    }
	  
	  // threadsafe end of block
	  pthread_mutex_lock(&mutex);
	  if ((block_count >= udpdb->packets_per_buffer) || (temp_idx >= temp_max))
	    {
	      syslog (LOG_INFO, "BLOCK COMPLETE thread_id=%d, seq_no=%"PRIu64", "
		      "ant_id=%"PRIu16", block_count=%"PRIu64", "
		      "temp_idx=%d, writeBlock=%d", thread_id, seq_no, ant_id,  block_count, 
		      temp_idx,writeBlock);

	      // write block
	      // check whether doWrite has been released. If not, skip this block
	      if (blockStatus[writeBlock % 64] > 0)
		blockStatus[writeBlock % 64] += 1;
	      else
		blockStatus[writeBlock % 64] = 1;
	      
	      uint64_t dropped = udpdb->packets_per_buffer - (block_count);
	      udpdb->packets->received += (block_count);
	      udpdb->bytes->received += (block_count) * UDP_DATA;	      
	      if (dropped)
		{
		  udpdb->packets->dropped += dropped;
		  udpdb->bytes->dropped += (dropped * UDP_DATA);
		}

	      // increment counters
	      dsaX_udpdb_increment(udpdb);
	      ctAnts = 0;

	      // write temp queue for this thread
	      //syslog(LOG_INFO,"thread %d: packets in this block %"PRIu64", temp_idx %d",thread_id,tpack,temp_idx);
	      tpack = 0;
	
	      for (i=0; i < temp_idx; i++)
		{
		  seq_byte = temp_seq_byte[i];
		  byte_offset = seq_byte - (block_start_byte);
		  if (byte_offset < udpdb->hdu_bufsz && byte_offset >= 0)
		    {
		      mod_WB = writeBlock % 64;
		      memcpy (udpdb->tblock + byte_offset + mod_WB*udpdb->hdu_bufsz, temp_buffers[i], UDP_DATA);
		      //pthread_mutex_lock(&mutex);
		      block_count++;		      
		      //pthread_mutex_unlock(&mutex);
		    }
		}
	      temp_idx = 0;
       
	    }
	  pthread_mutex_unlock(&mutex);

	  // at this stage, can try and write temp queue safely for other threads
	  if (temp_seq_byte[0] >= block_start_byte && temp_seq_byte[0] <= block_end_byte && temp_idx > 0)
	    {
	      //syslog(LOG_INFO,"thread %d: packets in this block %"PRIu64", temp_idx %d",thread_id,tpack,temp_idx);
	      tpack = 0;
	
	      for (i=0; i < temp_idx; i++)
		{
		  seq_byte = temp_seq_byte[i];
		  byte_offset = seq_byte - (block_start_byte);
		  if (byte_offset < udpdb->hdu_bufsz && byte_offset >= 0)
		    {
		      mod_WB = writeBlock % 64;
		      memcpy (udpdb->tblock + byte_offset + mod_WB*udpdb->hdu_bufsz, temp_buffers[i], UDP_DATA);
		      pthread_mutex_lock(&mutex);
		      block_count++;		      
		      pthread_mutex_unlock(&mutex);
		    }
		}
	      temp_idx = 0;

	    }

	}

      // packet has been inserted or saved by this point
      sock->have_packet = 0;
	
    }

  dsaX_free_sock(sock);
  free(temp_buffers);
  free(temp_seq_byte);
  
}

/* 
 *  Thread to write data
 */
void write_thread(void * arg) {

  dsaX_write_t * udpdb = (dsaX_write_t *) arg;
  int thread_id = udpdb->thread_id;

  // set affinity
  const pthread_t pid = pthread_self();
  int core_id;
  if (dPort==4011)
    core_id = write_cores[thread_id];
  else
    core_id = write_cores[thread_id+nwth];
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
    
  int mod_WB = 0;
  int a;
  
  while (!quit_threads)
  {

    mod_WB = lWriteBlock % 64;
    
    while (blockStatus[mod_WB]==0) {
      a=1;
    }    

    // assume everything is set up
    // wblock is assigned, write_ct=0
        
    memcpy(wblock + thread_id*udpdb->hdu_bufsz/nwth, udpdb->tblock + mod_WB*udpdb->hdu_bufsz  + thread_id*udpdb->hdu_bufsz/nwth, udpdb->hdu_bufsz/nwth);

    pthread_mutex_lock(&mutex);
    write_ct++;
    pthread_mutex_unlock(&mutex);

    //syslog(LOG_INFO,"write thread %d: successfully memcpied",thread_id);

    // now wait until thread 0 has finished getting a new block before moving on
    if (thread_id>0) {
      while (write_ct!=0) a=1;
    }
    else {

      // wait for all sub-blocks to be written
      while (write_ct<nwth) a=1;

      // get new block
      if (dsaX_udpdb_new_buffer (udpdb) < 0)
	{
	  syslog(LOG_ERR, "receive_obs: dsaX_udpdb_new_buffer failed");
	  return EXIT_FAILURE;
	}

      syslog(LOG_INFO,"write thread %d: written block... %d",thread_id,lWriteBlock);
      lWriteBlock++;
      
      // update doWrite and skipBlock
      skipct = 0;
      for (int i=0;i<64;i++) skipct += blockStatus[i];
      blockStatus[mod_WB] -= 1;
      write_ct = 0;

    }
     
  }

}


	    
// MAIN of program
	
int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_capture_manythread", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit for writing */
  dada_hdu_t* hdu_out = 0;
  
  // input data block HDU key
  key_t out_key = CAPTURE_BLOCK_KEY;

  // command line arguments
  int core = -1;
  int chgroup = 0;
  int arg=0;
  char dada_fnam[200]; // filename for dada header
  char iface[100]; // IP for data packets
  
  while ((arg=getopt(argc,argv,"c:j:i:f:o:g:p:q:dh")) != -1)
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
	case 'g':
	  if (optarg)
	    {	      
	      chgroup = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-g flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'j':
	  if (optarg)
	    {	      
	      strcpy(iface,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-j flag requires argument");
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
	case 'p':
	  if (optarg)
	    {
	      dPort = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-p flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }      	
	case 'q':
	  if (optarg)
	    {
	      cPort = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-q flag requires argument");
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
  udpdb_t temp_str;
  rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &temp_str);
  if (rval != 0) {
    syslog(LOG_ERR, "Error creating control_thread: %s", strerror(rval));
    return -1;
  }
  syslog(LOG_NOTICE, "Created control thread, listening on %s:%d",iP,cPort);
  
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
  dada_hdu_set_key (hdu_out, out_key);
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
     data are captured on iface:CAPTURE_PORT 
  */

  // make recv, write, and stats structs  
  udpdb_t udpdb[nth];
  dsaX_stats_t stats;
  dsaX_write_t writey[nwth];

  // shared variables and memory
  uint64_t bufsz = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  char * tblock = (char *)malloc(sizeof(char)*bufsz*64);
  stats_t * packets = init_stats_t();
  stats_t * bytes = init_stats_t();
  reset_stats_t(packets);
  reset_stats_t(bytes);

  // initialise stats struct
  stats.packets = packets;
  stats.bytes = bytes;

  // initialise writey struct and open buffer
  for (int i=0;i<nwth;i++) {
    writey[i].hdu = hdu_out;
    writey[i].hdu_bufsz = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
    writey[i].block_open = 0;
    writey[i].tblock = tblock;
    writey[i].thread_id = i;    
  }
  dsaX_udpdb_open_buffer (&writey[0]);

  // initialise all udpdb structs
  for (int i=0;i<nth;i++) {

    // shared stuff
    udpdb[i].packets = packets;
    udpdb[i].bytes = bytes;
    udpdb[i].tblock = tblock;

    // the rest
    udpdb[i].port = dPort;
    udpdb[i].interface = strdup(iface);
    udpdb[i].hdu_bufsz = bufsz;
    udpdb[i].packets_per_buffer = udpdb[i].hdu_bufsz / UDP_DATA;
    udpdb[i].num_inputs = NSNAPS;
    udpdb[i].verbose = 0;
    udpdb[i].rcv_sleeps = 0;
    
    udpdb[i].thread_id = i;    
    
  }


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
    rval = pthread_create (&recv_thread_id[i], 0, (void *) recv_thread, (void *) (&udpdb[i]));
    if (rval != 0) {
      syslog(LOG_ERR, "Error creating recv_thread %d: %s", i,strerror(rval));
      return -1;
    }
  }
  syslog(LOG_NOTICE, "Created recv threads");

  // start the write thread
  pthread_t write_thread_id[nwth];
  rval = 0;
  for (int i=0;i<nwth;i++) {
    rval = pthread_create (&write_thread_id[i], 0, (void *) write_thread, (void *) (&writey[i]));
    if (rval != 0) {
      syslog(LOG_INFO, "Error creating write_thread: %s", strerror(rval));
      return -1;
    }
  }
  syslog(LOG_NOTICE, "started write threads");  

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
  dsaX_dbgpu_cleanup (hdu_out);

}
