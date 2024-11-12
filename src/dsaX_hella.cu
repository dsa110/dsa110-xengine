// -*- c++ -*-

#include <time.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <dedisp.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <src/sigproc.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>
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

#include <cuda.h>
#include "cuda_fp16.h"
#include <cublas_v2.h>

//using namespace std;

const int MAXHOSTNAME = 200;
const int MAXCONNECTIONS = 5;
const int MAXRECV = 500;

class Socket
{
  public:
    Socket();
    virtual ~Socket();

    // Server initialization
    bool create();
    bool bind ( const char * address, const int port );
    bool listen() const;
    bool accept ( Socket& ) const;

    // Client initialization
    bool connect ( const std::string host, const int port );

    // Data Transimission
    bool send ( const std::string ) const;
    int recv ( std::string& ) const;

    void set_non_blocking ( const bool );

    bool is_valid() const { return m_sock != -1; }

    int select_timeout ( float sleep_secs );

  private:

    int m_sock;

    sockaddr_in m_addr;

};


class SocketException
{
  public:
    SocketException ( std::string s ) : m_s ( s ) {};
    ~SocketException (){};

    std::string description() { return m_s; }

  private:
    std::string m_s;

};


class ClientSocket : private Socket
{
  public:

    ClientSocket ( std::string host, int port );
    virtual ~ClientSocket(){};

    const ClientSocket& operator << ( const std::string& ) const;
    const ClientSocket& operator << ( const char * ) const;
    const ClientSocket& operator << ( const size_t n ) const;
    const ClientSocket& operator << ( const float f ) const;
    const ClientSocket& operator >> ( std::string& ) const;

};

Socket::Socket() :
  m_sock ( -1 )
{

  memset ( &m_addr, 0, sizeof ( m_addr ) );

}

Socket::~Socket()
{
  if ( is_valid() )
    ::close ( m_sock );
}

bool Socket::create()
{
  m_sock = socket ( AF_INET, SOCK_STREAM, 0 );

  if ( ! is_valid() )
    return false;


  // TIME_WAIT - argh
  int on = 1;
  if ( setsockopt ( m_sock, SOL_SOCKET, SO_REUSEADDR, ( const char* ) &on, sizeof ( on ) ) == -1 )
    return false;

  return true;
}


bool Socket::bind ( const char * address, const int port )
{

  if ( ! is_valid() )
  {
    return false;
  }

  m_addr.sin_family = AF_INET;
  m_addr.sin_port = htons ( port );
  m_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if (strcmp(address, "any") == 0)
    m_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  else
  {
    m_addr.sin_addr.s_addr = inet_addr (address);
    // if we didn't parse the address as an IP address
    if (m_addr.sin_addr.s_addr == -1)
    {
      struct hostent * hp = gethostbyname (address);
      memcpy(&(m_addr.sin_addr.s_addr), hp->h_addr, hp->h_length);
    }
  }

  int bind_return = ::bind ( m_sock,
           ( struct sockaddr * ) &m_addr,
           sizeof ( m_addr ) );


  if ( bind_return == -1 )
    return false;

  return true;
}


bool Socket::listen() const
{
  if ( ! is_valid() )
    return false;

  int listen_return = ::listen ( m_sock, MAXCONNECTIONS );

  if ( listen_return == -1 )
    return false;

  return true;
}


bool Socket::accept ( Socket& new_Socket ) const
{
  int addr_length = sizeof ( m_addr );
  new_Socket.m_sock = ::accept ( m_sock, ( sockaddr * ) &m_addr, ( socklen_t * ) &addr_length );

  if ( new_Socket.m_sock <= 0 )
    return false;
  else
    return true;
}

bool Socket::send ( const std::string s ) const
{
  int status = ::send ( m_sock, s.c_str(), s.size(), MSG_NOSIGNAL );
  if ( status == -1 )
    return false;
  else
    return true;
}

int Socket::recv ( std::string& s ) const
{
  char buf [ MAXRECV + 1 ];

  s = "";

  memset ( buf, 0, MAXRECV + 1 );

  int status = ::recv ( m_sock, buf, MAXRECV, 0 );

  if ( status == -1 )
  {
    std::cout << "status == -1   errno == " << errno << "  in Socket::recv\n";
    return 0;
  }
  else if ( status == 0 )
  {
    return 0;
  }
  else
  {
    s = buf;
    return status;
  }
}

bool Socket::connect ( const std::string address, const int port )
{
  if ( ! is_valid() ) return false;

  m_addr.sin_family = AF_INET;
  m_addr.sin_port = htons ( port );
  m_addr.sin_addr.s_addr = inet_addr (address.c_str());

  // if we didn't parse the address as an IP address
  if (m_addr.sin_addr.s_addr == -1)
  {
    struct hostent * hp = gethostbyname (address.c_str());
    memcpy(&(m_addr.sin_addr.s_addr), hp->h_addr, hp->h_length);
  }

  if ( errno == EAFNOSUPPORT ) return false;

  int status = ::connect ( m_sock, ( sockaddr * ) &m_addr, sizeof ( m_addr ) );

  if ( status == 0 )
    return true;
  else
    return false;
}

void Socket::set_non_blocking ( const bool b )
{
  int opts;

  opts = fcntl ( m_sock,
     F_GETFL );

  if ( opts < 0 )
    return;

  if ( b )
    opts = ( opts | O_NONBLOCK );
  else
    opts = ( opts & ~O_NONBLOCK );

  fcntl ( m_sock,
    F_SETFL,opts );

}

int Socket::select_timeout ( float sleep_secs )
{
  struct timeval timeout;
  fd_set *rdsp = NULL;
  fd_set readset;

  float whole_seconds = floor (sleep_secs);
  float micro_seconds = sleep_secs - whole_seconds;
  micro_seconds *= 100000;

  timeout.tv_sec = (long int) whole_seconds;
  timeout.tv_usec = (long int) micro_seconds;

  FD_ZERO (&readset);
  FD_SET (m_sock, &readset);
  rdsp = &readset;

  // select returns the number of file descriptors changed
  // 0 if timeout, -1 if error
  return select (m_sock+1, rdsp, NULL, NULL, &timeout);
}


ClientSocket::ClientSocket ( std::string host, int port )
{
  if ( ! Socket::create() )
    throw SocketException ( "Could not create client socket." );

  if ( ! Socket::connect ( host, port ) )
    throw SocketException ( "Could not bind to port." );

  //  printf("Connected to socket %d\n",port);
  
}


const ClientSocket& ClientSocket::operator << ( const std::string& s ) const
{
  if ( ! Socket::send ( s ) )
    throw SocketException ( "Could not write to socket." );

  return *this;
}

const ClientSocket& ClientSocket::operator << ( const char * s ) const
{
  if ( ! Socket::send ( std::string(s) ) )
    throw SocketException ( "Could not write to socket." );

  return *this;
}

const ClientSocket& ClientSocket::operator << ( const size_t n ) const
{
  std::ostringstream oss (std::ostringstream::out);
  oss << n;
  if ( ! Socket::send ( std::string( oss.str() ) ) )
    throw SocketException ( "Could not write to socket." );

  return *this;
}

const ClientSocket& ClientSocket::operator << ( const float f ) const
{
  std::ostringstream oss (std::ostringstream::out);
  oss << f;
  if ( ! Socket::send ( std::string( oss.str() ) ) )
    throw SocketException ( "Could not write to socket." );

  return *this;
}

const ClientSocket& ClientSocket::operator >> ( std::string& s ) const
{
  if ( ! Socket::recv ( s ) )
    throw SocketException ( "Could not read from socket." );

  return *this;
}


#define NMEDFILT 13
#define NTSMED 19
#define NBATCH 16
#define NCHAN 768
#define NBEAMS 64
#define NCHAN_BOX 48
#define NTIME_BOX 500
#define MAX_DM 2000
#define TOL 1.3
#define MAX_BOX 15
#define MAX_GIANTS 10000
#define DADA_BLOCK_KEY 0x0000dada // for capture program.

int finished = 0;

void dsaX_dbgpu_cleanup (dada_hdu_t * in);

void dsaX_dbgpu_cleanup (dada_hdu_t * in)
{

  if (dada_hdu_unlock_read (in) < 0)
    {
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);
  
}


// define a SCRUNCH structure
typedef struct scrunch {

  int tscrunch, fscrunch, nits;
  float thresh;

} scrunch;

// define a structure to carry all info and pre-allocated arrays, which can be passed between functions
typedef struct pinfo {

  // input params
  int inp_format; // 0 for dada, 1 for file, 2 for filterbank
  char inp_path[500];
  float minDM, maxDM, snr;
  int minWidth, maxWidth;
  int gulp;
  scrunch * scrunches; // array of scrunches
  int nscrunches;
  char beamflags[500], specflags[500];
  int out_format; // 0 for file, 1 for socket, 2 for both
  int coincidencer_port;
  std::string coincidencer_host;
  char out_path[500]; // path or IP
  int BEAM_OFFSET;
  int BEAM0;
  int flag1, flag2; // flag ranges
  
  // derived params
  int NTIME; // gulp that includes rewind
  int rewind; // samples to rewind by
  int nchan; // number of resampled channels
  int ndms; // number of DM trials
  int ntime_dd; // dedisp number of times
  int ntime_out; // final output number of times
  int ntime_dedisp; // number of dedispersed times (must be >= NTIME-max_delay)
  int nboxcar; 

  // pre-allocates - host
  unsigned char * data, * rewinds, * indata;
  float * h_dedisp;
  const float * DMs;
  dedisp_plan dedispersion_plan;
  float * h_dataF;
  float * h_flagSpec;
  
  // pre-allocates - GPU
  float * d_flagSpec;
  Npp32f * d_dedisp; // dedispersion output
  int d_dedisp_step;
  float * d_dedispPacked; // dedisp output
  unsigned char * d_inputPacked; // dedisp input
  unsigned char * d_data; // all the input  
  half * batch, * mask, * d_smooth;
  float * d_ts;
  int batch_stride;
  
  // boxcars
  Npp32f * boxes;
  int boxes_step;
  Npp32f * imbox;
  int imbox_step;
  float * stds;
  float mean;

  // peak finding
  thrust::device_vector<float> dmt;
  thrust::device_vector<int> output_indices;
  thrust::device_vector<float> output_values;
  int * h_idxs;
  int * beam, * out_beam;
  int * width, * out_width;
  int * dm_idx, * out_dm_idx; 
  int * samp, * out_samp;
  float * peaks, * out_peaks;
  int npeaks, out_npeaks;

  // flag timing
  float fcpy, fprep, fflag, fapply;
  float t1, t2, t3, t4, t5, t6, t7, t8, t9;
  
} pinfo;

int get_gpu_id(FILE *fconf) {

  char * line = NULL;
  ssize_t read;
  size_t len = 0;
  char c1[20], c2[500];

  while (!feof(fconf)) {

    read = getline(&line, &len, fconf);
    sscanf(line,"%s %s",c1,c2);
    if (strcmp(c1,"GPU")==0) {
      rewind(fconf);
      return atoi(c2);
    }

  }

}

// function to initialize everything based on a config file
void initialize(FILE *fconf, pinfo * p) {

  // read input file line by line
  p->nscrunches = 0;
  p->BEAM_OFFSET = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;
  char c1[20], c2[500];
  p->flag1 = -1;
  p->flag2 = -1;
  while (!feof(fconf)) {

    read = getline(&line, &len, fconf);
    sscanf(line,"%s %s",c1,c2);

    if (strcmp(c1,"INPUT")==0) {
      if (strcmp(c2,"DADA")==0) p->inp_format=0;
      if (strcmp(c2,"FILE")==0) p->inp_format=1;
      if (strcmp(c2,"FILTERBANK")==0) p->inp_format=2;
      printf("Using input format %d\n",p->inp_format);
    }
    if (strcmp(c1,"OUTPUT")==0) {
      if (strcmp(c2,"FILE")==0) p->out_format=0;
      if (strcmp(c2,"SOCKET")==0) p->out_format=1;
      if (strcmp(c2,"BOTH")==0) p->out_format=2;
      printf("Using output format %d\n",p->out_format);
    }
    if (strcmp(c1,"HOST")==0) {
      p->coincidencer_host = c2;
    }
    if (strcmp(c1,"BEAM0")==0) {
      p->BEAM0 = atoi(c2);
    }
    if (strcmp(c1,"PORT")==0) {
      p->coincidencer_port = atoi(c2);
    }    
    if (strcmp(c1,"OUTPUTPATH")==0) {
      strcpy(p->out_path,c2);
    }
    if (strcmp(c1,"BEAM_OFFSET")==0) {
      p->BEAM_OFFSET = atoi(c2);
    }
      
    if (strcmp(c1,"BEAMFLAGS")==0) {
      strcpy(p->beamflags,c2);
    }
    if (strcmp(c1,"SPECFLAGS")==0) {
      strcpy(p->specflags,c2);
    }

    if (strcmp(c1,"INPUT_PATH")==0) {
      strcpy(p->inp_path,c2);
      printf("Input path: %s\n",p->inp_path);
    }

    if (strcmp(c1,"DM_MIN")==0)
      p->minDM=atof(c2);
    if (strcmp(c1,"DM_MAX")==0)
      p->maxDM=atof(c2);
    if (strcmp(c1,"WIDTH_MIN")==0) {
      p->minWidth=atoi(c2);
    }
    if (strcmp(c1,"WIDTH_MAX")==0) {
      p->maxWidth=atoi(c2);
    }
    if (strcmp(c1,"SNR")==0) {
      p->snr=atof(c2);
    }
    if (strcmp(c1,"GULP")==0)
      p->gulp=atoi(c2);
    if (strcmp(c1,"FLAG1")==0)
      p->flag1=atoi(c2);
    if (strcmp(c1,"FLAG2")==0)
      p->flag2=atoi(c2);
    
    if (strcmp(c1,"SCRUNCH")==0) {

      p->nscrunches = atoi(c2);
      p->scrunches = (scrunch *)malloc(p->nscrunches*sizeof(scrunch));

      for (int i=0;i<p->nscrunches;i++) {
	read = getline(&line, &len, fconf);
	sscanf(line,"%d %d %f %d",&(p->scrunches[i].tscrunch),&(p->scrunches[i].fscrunch),&(p->scrunches[i].thresh),&(p->scrunches[i].nits));
	printf("Have a scrunch with %d %d %g %d\n",p->scrunches[i].tscrunch,p->scrunches[i].fscrunch,p->scrunches[i].thresh,p->scrunches[i].nits);
      }
      
    }   
  }
  fclose(fconf);

  // derived parameters
  p->NTIME=p->gulp;
  p->rewind=0;
  int i=(int)(p->minWidth), j=0;
  while (i<(int)(p->maxWidth))  {
    i *= 2;
    j += 1;
  }
  p->nboxcar=j;
  printf("Search parameters: DM range %g to %g, WIDTHS %d to %d (%d trials), SNR %g\n",p->minDM,p->maxDM,p->minWidth,p->maxWidth,p->nboxcar,p->snr);
  if (p->out_format != 0)
    printf("Outputting to socket %s:%d\n",p->coincidencer_host.c_str(),p->coincidencer_port);
  if (p->out_format != 1)
    printf("Outputting to text file %s\n",p->out_path);
  
  
  // set up DM plan
  dedisp_create_plan(&p->dedispersion_plan,NCHAN,262.144e-6,1498.75,0.244140625);
  // generate DM list  
  dedisp_generate_dm_list(p->dedispersion_plan,p->minDM,p->maxDM,40,TOL);
  p->DMs = dedisp_get_dm_list(p->dedispersion_plan);
  p->ndms = dedisp_get_dm_count(p->dedispersion_plan);    
  p->ntime_dd = p->NTIME - dedisp_get_max_delay(p->dedispersion_plan);
  p->ntime_out = p->ntime_dd - p->maxWidth;
  p->ntime_dedisp = p->ntime_dd;
  // modify NTIME and ntime_dd in case of non-text input
  int oo;
  if (p->inp_format != 1) {
    p->NTIME = p->gulp + dedisp_get_max_delay(p->dedispersion_plan) + p->maxWidth;
    oo = 32*((int)(p->NTIME/32)+1);
    p->NTIME = oo;
    p->ntime_dedisp = oo-dedisp_get_max_delay(p->dedispersion_plan);
    p->ntime_dd = p->gulp + p->maxWidth;
    p->ntime_out = p->gulp;
  }

  
  // allocate everything

  p->h_flagSpec = (float *)malloc(sizeof(float)*NCHAN*NBATCH);
  cudaMalloc((void **)(&p->d_flagSpec), sizeof(float)*NCHAN*NBATCH);
  p->rewinds = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*(p->NTIME-p->gulp)*NBEAMS);
  memset(p->rewinds,0,NCHAN*(p->NTIME-p->gulp)*NBEAMS);  
  p->data = (unsigned char *)malloc(sizeof(unsigned char)*NBEAMS*NCHAN*p->NTIME);
  //p->h_dedisp = (float *)malloc(sizeof(float)*p->ndms*p->ntime_dd);
  //p->indata = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*p->NTIME);
  p->h_dataF = (float *)malloc(sizeof(float)*NCHAN*p->NTIME);
  cudaMalloc((void **)(&p->d_dedispPacked), sizeof(float)*p->ndms*p->ntime_dedisp);
  cudaMalloc((void **)(&p->d_inputPacked), sizeof(unsigned char)*NCHAN*p->NTIME);
  cudaMalloc((void **)(&p->d_data), sizeof(unsigned char)*NBEAMS*NCHAN*p->NTIME);
  cudaMallocPitch((void **)(&p->batch), (size_t *)(&p->batch_stride), (unsigned long)(p->NTIME*sizeof(half)), NBATCH*NCHAN);
  cudaMallocPitch((void **)(&p->mask), (size_t *)(&p->batch_stride), (unsigned long)(p->NTIME*sizeof(half)), NBATCH*NCHAN);
  p->batch_stride = p->batch_stride / sizeof(half);
  cudaMalloc((void **)(&p->d_smooth), NBATCH * NCHAN * p->batch_stride * sizeof(half));  
  p->d_dedisp = nppiMalloc_32f_C1(p->ntime_dd,p->ndms,&(p->d_dedisp_step));
  cudaMalloc((void **)(&p->d_ts), NBATCH * p->NTIME * sizeof(float));

  printf("Will use %d DM trials, output %d times, process %d times with stride %d\n",p->ndms,p->ntime_dd,p->NTIME,p->batch_stride);

  
  // boxcars
  p->boxes = nppiMalloc_32f_C1(p->ntime_out,(p->ndms-2)*p->nboxcar,&(p->boxes_step));
  p->imbox = nppiMalloc_32f_C1(p->ntime_dd,p->ndms,&(p->imbox_step));
  p->stds = (float *)malloc(sizeof(float)*p->nboxcar);
  p->mean = 0.21368;
  p->stds[0] = 0.001309;
  p->stds[1] = 0.00124735;
  p->stds[2] = 0.00103835;
  p->stds[3] = 0.00081225;
  p->stds[4] = 0.00062605;
  p->stds[5] = 0.00047785;
  p->stds[6] = 0.00036005;
  
  // peak finding
  p->dmt.resize((p->ndms-2)*p->ntime_out);
  p->output_indices.resize((p->ndms-2)*p->ntime_out);
  p->output_values.resize((p->ndms-2)*p->ntime_out);
  p->h_idxs = (int *)malloc(sizeof(int)*(p->ndms-2)*p->ntime_out);
  p->beam = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->width =  (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->dm_idx = (int *)malloc(sizeof(int)*MAX_GIANTS); 
  p->samp = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->peaks = (float *)malloc(sizeof(float)*MAX_GIANTS);

  p->out_beam = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_samp = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_width =  (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_dm_idx = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_peaks = (float *)malloc(sizeof(float)*MAX_GIANTS);
  
  // flag timing
  p->fcpy=0.;
  p->fprep=0.;
  p->fflag=0.;
  p->fapply=0.;
  p->t1=0.;
  p->t2=0.;
  p->t3=0.;
  p->t4=0.;
  p->t5=0.;
  p->t6=0.;
  p->t7=0.;
  p->t8=0.;
  p->t9=0.;
  
}

// deallocate everything
void deallocator(pinfo * p) {

  printf("deallocating pinfo struct\n");
  free(p->data);
  free(p->h_dataF);
  cudaFree(p->d_data);
  cudaFree(p->batch);
  cudaFree(p->mask);
  cudaFree(p->d_dedisp);
  cudaFree(p->boxes);
  cudaFree(p->d_dedispPacked);
  cudaFree(p->d_inputPacked);
  p->dmt.clear();
  p->output_indices.clear();
  p->output_values.clear();
  free(p->h_idxs);
  free(p->beam);
  free(p->width);
  free(p->dm_idx);
  free(p->samp);
  free(p->peaks);
  free(p->stds);
  free(p->rewinds);
  
}

void help() {

  printf("Usage: pipeline -c <config file>\n");
  printf("Everything is in the config file. Specific parameters include: \n");
  printf("INPUT <DADA or FILE or FILTERBANK>\n");
  printf("INPUT_PATH <dada buffer or full path to filterbank file>\n");
  printf("BEAM_OFFSET <offset in number of beams in input dada buffer>\n");
  printf("DM_MIN <min DM of search>\n");
  printf("DM_MAX <max DM of search>\n");
  printf("WIDTH_MIN <min width of search>\n");
  printf("WIDTH_MAX <max width of search>\n");
  printf("SNR <SNR threshold for search>\n");
  printf("GULP <base gulp size>\n");
  printf("BEAMFLAGS <full path to beam flags output>\n");
  printf("SPECFLAGS <full path to spec flags output>\n");
  printf("OUTPUT <FILE or SOCKET of BOTH>\n");
  printf("OUTPUTPATH <path to output file>\n");
  printf("HOST <ip of T2 host>\n");
  printf("PORT <T2 port>\n");
  printf("GPU <GPU ID 0 or 1>\n");
  printf("BEAM0 <first beam in output>\n");
  printf("SCRUNCH <number of scrunches>\n");
  printf("<time scrunch> <frequency scrunch> <flagging threshold> <number of iterations>\n");
  printf("repeat the above as many times as you like for different parameters\n");

}


/************ FUNCTIONS FOR NEW FLAGGER ***********/

// kernel to warp reduce float
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

// kernel to sum array and its squares
__global__ void sumArray(half * data, float * sums, float * qsums, int width, int height, int stride) {

  extern __shared__ float sdata[512], qdata[512];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*512 + tid;
  int x = i % width;
  int y = i / width;
  int iidx = y*stride+x;

  sdata[tid] = __half2float(data[iidx]);
  qdata[tid] = __half2float(data[iidx]*data[iidx]);

  __syncthreads();

  if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); 
  if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); 
  if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); 
  if (tid < 32) warpReduce(sdata, tid);
  if (tid < 256) { qdata[tid] += qdata[tid + 256]; } __syncthreads(); 
  if (tid < 128) { qdata[tid] += qdata[tid + 128]; } __syncthreads(); 
  if (tid < 64) { qdata[tid] += qdata[tid + 64]; } __syncthreads(); 
  if (tid < 32) warpReduce(qdata, tid);

  if (tid == 0) sums[blockIdx.x] = sdata[0];
  if (tid == 0) qsums[blockIdx.x] = qdata[0];
  

}
// kernel to sum array and its squares
__global__ void sumArrayFloat(float * data, float * sums, float * qsums, int width, int height, int stride) {

  extern __shared__ float sfdata[512], qfdata[512];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*512 + tid;
  int x = i % width;
  int y = i / width;
  int iidx = y*stride+x;

  sfdata[tid] = data[iidx];
  qfdata[tid] = data[iidx]*data[iidx];

  __syncthreads();

  if (tid < 256) { sfdata[tid] += sfdata[tid + 256]; } __syncthreads(); 
  if (tid < 128) { sfdata[tid] += sfdata[tid + 128]; } __syncthreads(); 
  if (tid < 64) { sfdata[tid] += sfdata[tid + 64]; } __syncthreads(); 
  if (tid < 32) warpReduce(sfdata, tid);
  if (tid < 256) { qfdata[tid] += qfdata[tid + 256]; } __syncthreads(); 
  if (tid < 128) { qfdata[tid] += qfdata[tid + 128]; } __syncthreads(); 
  if (tid < 64) { qfdata[tid] += qfdata[tid + 64]; } __syncthreads(); 
  if (tid < 32) warpReduce(qfdata, tid);

  if (tid == 0) sums[blockIdx.x] = sfdata[0];
  if (tid == 0) qsums[blockIdx.x] = qfdata[0];
  

}
    

// Host function to orchestrate the normalization process
float calculateStdDev(half * d_data, int width, int height, int stride) {

  float *d_sums, *d_qsums;
  int new_width = (int)(512*floor(width/512.));
  int nblocks = new_width*height / 512;
  
  cudaMalloc(&d_sums, nblocks * sizeof(float));
  cudaMalloc(&d_qsums, nblocks * sizeof(float));
  sumArray<<<nblocks,512>>>(d_data,d_sums,d_qsums,new_width,height,stride);

  float *sums, *qsums;
  sums = (float *)malloc(sizeof(float)*nblocks);
  qsums = (float *)malloc(sizeof(float)*nblocks);
  cudaMemcpy(sums,d_sums,nblocks*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(qsums,d_qsums,nblocks*sizeof(float),cudaMemcpyDeviceToHost);

  float sum=0., qsum=0.;
  for (int i=0;i<nblocks;i++) {
    sum += sums[i];
    qsum += qsums[i];
  }
  float mn = sum/(new_width*height*1.);

  float stdDev = qsum-2.*sum*mn+mn*mn*new_width*height*1.;
  stdDev /= 1.*new_width*height;
  stdDev = sqrt(stdDev);

  cudaFree(d_sums);
  cudaFree(d_qsums);
  free(sums);
  free(qsums);
  
  return stdDev;
  
  
  
}
// Host function to orchestrate the normalization process
float calculateStdDevFloat(float * d_data, int width, int height, int stride) {

  float *d_sums, *d_qsums;
  int new_width = (int)(512*floor(width/512.));
  int nblocks = new_width*height / 512;
  
  cudaMalloc(&d_sums, nblocks * sizeof(float));
  cudaMalloc(&d_qsums, nblocks * sizeof(float));
  sumArrayFloat<<<nblocks,512>>>(d_data,d_sums,d_qsums,new_width,height,stride);

  float *sums, *qsums;
  sums = (float *)malloc(sizeof(float)*nblocks);
  qsums = (float *)malloc(sizeof(float)*nblocks);
  cudaMemcpy(sums,d_sums,nblocks*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(qsums,d_qsums,nblocks*sizeof(float),cudaMemcpyDeviceToHost);

  float sum=0., qsum=0.;
  for (int i=0;i<nblocks;i++) {
    sum += sums[i];
    qsum += qsums[i];
  }
  float mn = sum/(new_width*height*1.);

  float stdDev = qsum-2.*sum*mn+mn*mn*new_width*height*1.;
  stdDev /= 1.*new_width*height;
  stdDev = sqrt(stdDev);

  cudaFree(d_sums);
  cudaFree(d_qsums);
  free(sums);
  free(qsums);
  
  return stdDev;
  
  
  
}


// warp reduce half
/*__device__ void warpReduceHalf(volatile half *data, int tid) {
  data[tid] = data[tid] + data[tid + 32];
  data[tid] = data[tid] + data[tid + 16];
  data[tid] = data[tid] + data[tid + 8];
  data[tid] = data[tid] + data[tid + 4];
  data[tid] = data[tid] + data[tid + 2];
  data[tid] = data[tid] + data[tid + 1];
  }*/

// add bandpasses
// add_bandpass<<<NCHAN*NBATCH/32,32>>>(d_mask,d_flagSpec);
__global__ void add_bandpass(float * d_mask, float * d_flagSpec) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx = bid*32 + tid;

  d_flagSpec[idx] += d_mask[idx];

}
  
// kernel to calculate bandpass
// launch with NCHAN*NBATCH blocks of 256 threads
// will ignore last (width % 256) times
__global__ void calc_bandpass(half * data, float * bandpass, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  
  int npartials = (int)(width / 256); // number partial sums
  half fac = (half)((256.*npartials));
  extern __shared__ half psum[256];

  // calculate partial sums
  int idx0 = bid*stride + tid*npartials;
  psum[tid] = 0.;
  for (int i=idx0;i<npartials+idx0;i++)
    psum[tid] += data[i];
  psum[tid] /= fac;

  __syncthreads();

  // sum over shared memory
  if (tid < 128) { psum[tid] += psum[tid + 128]; } __syncthreads(); 
  if (tid < 64) { psum[tid] += psum[tid + 64]; } __syncthreads();
  if (tid < 32) { psum[tid] += psum[tid + 32]; } __syncthreads();
  if (tid < 16) { psum[tid] += psum[tid + 16]; } __syncthreads();
  if (tid < 8) { psum[tid] += psum[tid + 8]; } __syncthreads();
  if (tid < 4) { psum[tid] += psum[tid + 4]; } __syncthreads();
  if (tid < 2) { psum[tid] += psum[tid + 2]; } __syncthreads();
  if (tid < 1) { psum[tid] += psum[tid + 1]; } __syncthreads(); 
  //if (tid < 32) warpReduce(psum, tid);
  __syncthreads();

  if (tid==0) bandpass[bid] = __half2float(psum[0]);
  
}

// Function to swap two float values
void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

// Function to find the median of an array
float findMedian(float arr[], int n) {
    // Sort the array using bubble sort (you can use faster algorithms)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
    
    // Return the middle element if odd, or average of two middle elements if even
    if (n % 2 != 0) 
        return (float)arr[n / 2];
    return (float)(arr[(n - 1) / 2] + arr[n / 2]) / 2.0;
}

// Function to apply median filter
float medianFilter(float *input, float *output, int size, int windowSize) {
  
    if (windowSize % 2 == 0) {
        printf("Error: Window size must be odd.\n");
        return 0.; 
    }

    int halfWindowSize = windowSize / 2;

    for (int i = halfWindowSize; i < size-halfWindowSize; i++) {
        float window[windowSize];
        int windowIndex = 0;

        // Populate the window array
        for (int j = i - halfWindowSize; j <= i + halfWindowSize; j++) {
            if (j >= 0 && j < size) {
                window[windowIndex] = input[j];
            } else {
                // Handle boundary cases (you can choose different strategies)
                window[windowIndex] = input[i]; // Repeat edge value
            }
            windowIndex++;
        }

        // Calculate and store the median for the current window
        output[i] = findMedian(window, windowSize);
    }

    // edge values
    for (int i=0;i<halfWindowSize;i++) output[i] = output[halfWindowSize];
    for (int i=size-halfWindowSize;i<size;i++) output[i] = output[size-halfWindowSize-1];

    // return mean
    float mn = 0.;
    for (int i=0;i<size;i++)
      mn += output[i];
    mn /= 1.*size;

    return mn;
    
}

// function to orchestrate host median filtering of bandpass
float medFilterBandpass(float * d_bandpass) {

  float * hbp = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
  float * mhbp = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
  cudaMemcpy(hbp,d_bandpass,sizeof(float)*NBATCH*NCHAN,cudaMemcpyDeviceToHost);
  float mn_bp = medianFilter(hbp,mhbp,NBATCH*NCHAN,NMEDFILT);
  cudaMemcpy(d_bandpass,mhbp,sizeof(float)*NBATCH*NCHAN,cudaMemcpyHostToDevice);

  free(hbp);
  free(mhbp);

  return mn_bp;
  
}

// function to orchestrate host median filtering of time series
void medFilterTs(float * d_ts, int width) {

  float * hts = (float *)malloc(sizeof(float)*NBATCH*width);
  float * mhts = (float *)malloc(sizeof(float)*NBATCH*width);
  cudaMemcpy(hts,d_ts,sizeof(float)*NBATCH*width,cudaMemcpyDeviceToHost);
  float mn_ts = medianFilter(hts,mhts,NBATCH*width,NTSMED);
  cudaMemcpy(d_ts,mhts,sizeof(float)*NBATCH*width,cudaMemcpyHostToDevice);

  free(hts);
  free(mhts);
  
}

// cuda kernel to divide data by bandpass
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void divide_by_bp(half * data, float * bp, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] /= __float2half(bp[y]);

}

// cuda kernel to divide data by time series
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void divide_by_ts(half * data, float * ts, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int b = (int)(y / NCHAN);
  int tsidx = b*width+x;
  int iidx = y*stride+x;

  data[iidx] /= __float2half(ts[tsidx]);

}

// handler for half-precision boxcar convolution from npp
void npp_convolve_handler(half * data, half * output, float scfac, int xw, int yw, int width, int stride) {

  NppiSize oSrcSize = {stride,NCHAN};
  NppiPoint oSrcOffset = {0,0};
  Npp32f * pKernel;
  NppiSize pKernelSize = {xw,yw};
  NppiPoint oAnchor = {(int)(xw/2),(int)(yw/2)};  
  float * h_kernel;
  cudaMalloc((void **)(&pKernel), 4*xw*yw);
  h_kernel  = (float *)malloc(sizeof(float)*xw*yw);
  for (int i=0;i<xw*yw;i++) h_kernel[i] = scfac;
  cudaMemcpy(pKernel,h_kernel,4*xw*yw,cudaMemcpyHostToDevice);

  nppiFilterBorder32f_16f_C1R((Npp16f *)data,stride*2,oSrcSize,oSrcOffset,(Npp16f *)output,stride*2,oSrcSize,pKernel,pKernelSize,oAnchor,NPP_BORDER_REPLICATE);

  cudaFree(pKernel);
  free(h_kernel);
  
}


// cuda kernel to smooth 2d array and output into different array
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void smooth_data(half * data, half * output, float scfac, int xw, int yw, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  half mysc = __float2half(scfac);

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;
  int ch = (int)(y % NCHAN);
  int bt = (int)(y / NCHAN);

  int xs = x-xw/2;
  int xe = x+xw/2+1;
  if (xs<0) {
    xe -= xs;
    xs = 0;
  }
  else if (xe>width) {
    xs -= xe-width;
    xe = width;
  }

  int ys = ch - yw/2;
  int ye = ch + yw/2 + 1;
  if (ys<0) {
    ye -= ys;
    ys = 0;
  }
  else if (ye>NCHAN) {
    ys -= ye-NCHAN;
    ye = NCHAN;
  }

  output[iidx] = 0.;
  for (int yi=ys; yi<ye; yi++) {
    for (int xi=xs; xi<xe; xi++) 
      output[iidx] += data[bt*NCHAN*stride + yi*stride + xi]*mysc;
  }
  //output[iidx] *= mysc;

}


// cuda kernel to divide data by array
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void divide_by_array(half * data, half * arr, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] /= arr[iidx];

}

// cuda kernel to multiply data by number
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void multiply_by_number(half * data, float num, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] *= __float2half(num);

}

// cuda kernel to add number to data
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void add_number(half * data, float num, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] += __float2half(num);

}

// cuda kernel to find data above threshold and add into mask
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void threshold_data(half * data, half * mask, float threshold, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  if (data[iidx]>__float2half(threshold))
    mask[iidx] = 1.;

}

// cuda kernel to replace masked values
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void replace_data(half * data, half * mask, float repval, int width, int stride, int flag1, int flag2) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int ch = (int)(y % NCHAN);
  int iidx = y*stride+x;

  if (mask[iidx]>(half)(0.))
    data[iidx] = repval;

  if (ch>=flag1 && ch<flag2)
    data[iidx] = repval;

}

// cuda kernel to transpose and scale single beam data for loader
// beam is [width, NCHAN], data is [NCHAN, width]
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(NCHAN/32, width/32)
__global__ void transpose_input(unsigned char * beam, half * data, int width) {

  __shared__ half tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int mywidth = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = __float2half((float)(beam[(y+j)*mywidth + x]));

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  mywidth = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8)
    data[(y+j)*mywidth + x] = tile[threadIdx.x][threadIdx.y + j];
  
}

// cuda kernel to transpose and scale single beam data for output
// beam is [width, NCHAN], data is [NCHAN, width] 
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(width/32, NCHAN/32)
__global__ void transpose_output(unsigned char * beam, half * data, int width) {

  __shared__ half tile[32][33];
  
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int mywidth = gridDim.x * 32;
  
  for (int j = 0; j < 32; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = data[(y+j)*mywidth + x];

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  mywidth = gridDim.y * 32;

  // to saturate uchar8 at -4. and 10.
  float v, scf;
  for (int j = 0; j < 32; j += 8) {
    v = __half2float(tile[threadIdx.x][threadIdx.y + j]);
    scf = 255./14.;
    v = scf*(v+4.);
    if (v<0.) v = 0;
    if (v>255.) v = 255.;
    beam[(y+j)*mywidth + x] = (unsigned char)(v);
  }

}

// TODO: handle_transpose_input and handle_transpose_output to do transpose via memcpy2d from intermediate array

void transpose_input_handler(unsigned char * d_data, half * batch, int width, int stride) {

  dim3 dimBlockIn(32, 8), dimGridIn(NCHAN/32, width/32);
  half * tmpBuffer;
  cudaMalloc(&tmpBuffer, NCHAN * width * sizeof(half));

  // do transpose by beam
  for (int bm=0; bm<NBATCH; bm++) {
    transpose_input<<<dimGridIn,dimBlockIn>>>(d_data+bm*NCHAN*width,tmpBuffer,width);
    cudaMemcpy2D(batch+bm*NCHAN*stride,stride*sizeof(half),tmpBuffer,width*sizeof(half),width*sizeof(half),NCHAN,cudaMemcpyDeviceToDevice);
  }

  cudaFree(tmpBuffer);

}

void transpose_output_handler(unsigned char * d_data, half * batch, int width, int stride) {

  dim3 dimBlockOut(32, 8), dimGridOut(width/32, NCHAN/32);
  half * tmpBuffer;
  cudaMalloc(&tmpBuffer, NCHAN * width * sizeof(half));

  // do transpose by beam
  for (int bm=0; bm<NBATCH; bm++) {
    cudaMemcpy2D(tmpBuffer,width*sizeof(half),batch+bm*NCHAN*stride,stride*sizeof(half),width*sizeof(half),NCHAN,cudaMemcpyDeviceToDevice);
    transpose_output<<<dimGridOut,dimBlockOut>>>(d_data+bm*NCHAN*width,tmpBuffer,width);
  }

  cudaFree(tmpBuffer);
  
}

// kernel to sort out time series
// run with width*NBATCH/32 blocks of 32 threads
__global__ void fix_ts(half * temp_ts, float * ts, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx = bid*32+tid;
  int bat = (int)(idx/NBATCH);
  int tim = (int)(idx % NBATCH);

  ts[idx] = __half2float(temp_ts[bat*stride+tim]);
  
}

// function to measure ts using cublas calls
void blas_ts(half * data, half * unity, half * temp_ts, float * ts, int width, int stride) {

  // set up for gemm
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  // gemm settings
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = stride;
  const int n = 1;
  const int k = NCHAN;
  const half alpha = 1./NCHAN;
  const int lda = m;
  const int ldb = k;
  const half beta = 0.;
  const int ldc = m;
  const long long int strideA = NCHAN*stride;
  const long long int strideB = NCHAN;
  const long long int strideC = stride;
  const int batchCount = NBATCH;

  // run strided batched gemm
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,data,lda,strideA,
			    unity,ldb,strideB,&beta,
			    temp_ts,ldc,strideC,
			    batchCount);

  cudaDeviceSynchronize();
  
  // run kernel to place in output
  fix_ts<<<width*NBATCH/32,32>>>(temp_ts,ts,width,stride);


}

// cuda kernel to measure per-beam time baseline
// run with NBATCH*width lots of 32 threads
__global__ void measure_ts(half * data, float * ts, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int id = bid*32+tid;
  int iTime = (int)(bid % width);
  int iBatch = (int)(bid / width);
  
  int npartials = (int)(NCHAN / 32); // number partial sums
  half fac = (half)((32.*npartials));
  extern __shared__ half cpsum[32];

  // calculate partial sums
       
  cpsum[tid] = 0.;
  int idx0 = iBatch*NCHAN*stride + tid*npartials*stride + iTime;
  for (int i=idx0;i<idx0+npartials*stride;i+=stride)
    cpsum[tid] += data[i];
  cpsum[tid] /= fac;

  __syncthreads();

  // sum over shared memory
  if (tid < 16) { cpsum[tid] += cpsum[tid + 16]; } __syncthreads();
  if (tid < 8) { cpsum[tid] += cpsum[tid + 8]; } __syncthreads();
  if (tid < 4) { cpsum[tid] += cpsum[tid + 4]; } __syncthreads();
  if (tid < 2) { cpsum[tid] += cpsum[tid + 2]; } __syncthreads();
  if (tid < 1) { cpsum[tid] += cpsum[tid + 1]; } __syncthreads(); 
  //if (tid < 32) warpReduce(psum, tid);
  __syncthreads();

  if (tid==0) ts[bid] = __half2float(cpsum[0]);
  
}

// function to bandpass-correct data
float bandpass_correct(half * data, int width, int stride) {

  // allocate bandpass
  float * d_bandpass;
  cudaMalloc(&d_bandpass, NBATCH * NCHAN * sizeof(float));

  // calculate bandpass
  calc_bandpass<<<NCHAN*NBATCH,256>>>(data, d_bandpass, width, stride);

  cudaDeviceSynchronize();
    
  // median filter bandpass
  float mn_bp = medFilterBandpass(d_bandpass);
  
  // correct bandpass in data
  divide_by_bp<<<NBATCH*NCHAN*width/32,32>>>(data,d_bandpass,width,stride);

  cudaFree(d_bandpass);

  return mn_bp;
  
}

// function to ts-correct data
void ts_correct(half * data, float * d_ts, int width, int stride) {

  // calculate ts
  measure_ts<<<NBATCH*width,32>>>(data, d_ts, width, stride);

  cudaDeviceSynchronize();
    
  // median filter ts
  //medFilterTs(d_ts,width);
  
  // correct ts in data
  divide_by_ts<<<NBATCH*NCHAN*width/32,32>>>(data,d_ts,width,stride);

}


// function to remove time-frequency baseline
void remove_tf_baseline(half * data, int width, int stride) {

  // allocate smooth array
  half * d_smooth;
  cudaMalloc(&d_smooth, NBATCH * NCHAN * stride * sizeof(half));

  // smooth data
  //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, (float)(1./NTIME_BOX/NCHAN_BOX), NTIME_BOX, NCHAN_BOX, width, stride);
  npp_convolve_handler(data, d_smooth, (float)(1./NTIME_BOX/NCHAN_BOX), NTIME_BOX, NCHAN_BOX, width, stride);

  // divide by smoothed data
  divide_by_array<<<NBATCH*NCHAN*width/32,32>>>(data,d_smooth,width,stride);  

  cudaFree(d_smooth);
  
}

// function to normalize data
// assume initial mean is 1
void normalize_data(half * data, int width, int stride) {

  float stdDev = calculateStdDev(data,width,NBATCH*NCHAN,stride);

  add_number<<<NBATCH*NCHAN*width/32,32>>>(data,-1.,width,stride);

  multiply_by_number<<<NBATCH*NCHAN*width/32,32>>>(data,1./stdDev,width,stride);

}

// function to apply a single scrunch to the data
float apply_scrunch(pinfo * p, half * data, half * mask, half * d_smooth, float * d_ts, int width, int stride, int tscrunch, int fscrunch, float thresh, int flag, int ts, float * d_flagSpec, int flag1, int flag2) {

  float begin, end;
  float * d_mask;
  cudaMalloc(&d_mask, NBATCH * NCHAN * sizeof(float));
  
  // bandpass
  //printf("bandpass\n");
  begin = clock();
  float mn_bp = bandpass_correct(data,width,stride);
  cudaDeviceSynchronize();
  end = clock();
  p->t1 += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // baseline
  //printf("baseline\n");
  if (ts==1) {
    begin = clock();
    ts_correct(data,d_ts,width,stride);
    cudaDeviceSynchronize();
    end = clock();
    p->t2 += (float)(end - begin) / CLOCKS_PER_SEC;
  }

  // normalize
  //printf("normalize\n");
  begin = clock();
  normalize_data(data,width,stride);
  cudaDeviceSynchronize();
  end = clock();
  p->t3 += (float)(end - begin) / CLOCKS_PER_SEC;

  if (flag==1) {
    
    // derive smoothed data
    //printf("smooth\n");
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    npp_convolve_handler(data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    cudaDeviceSynchronize();
    end = clock();
    p->t4 += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // threshold data
    //printf("threshold\n");
    begin = clock();
    cudaMemset(mask,0,NBATCH*NCHAN*stride*sizeof(half));
    threshold_data<<<NBATCH*NCHAN*width/32,32>>>(d_smooth,mask,thresh/sqrt(1.*tscrunch*fscrunch),width,stride);
    cudaDeviceSynchronize();
    end = clock();
    p->t5 += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // replace data after growing mask
    //printf("replace\n");
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    npp_convolve_handler(mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    replace_data<<<NBATCH*NCHAN*width/32,32>>>(data, mask, 0., width, stride, flag1, flag2);
    add_number<<<NBATCH*NCHAN*width/32,32>>>(data,1.,width,stride);
    calc_bandpass<<<NCHAN*NBATCH,256>>>(mask, d_mask, width, stride);
    add_bandpass<<<NCHAN*NBATCH/32,32>>>(d_mask,d_flagSpec);
    cudaDeviceSynchronize();
    end = clock();
    p->t6 += (float)(end - begin) / CLOCKS_PER_SEC;

  }

  cudaFree(d_mask);

  return mn_bp;
  
}

// flagger
// load a batch beam by beam
// apply all scrunches to the batch
// unload the batch
void fastflagger(pinfo * p) {

  float begin, end;
  
  // setup
  int nBatches = (int)(NBEAMS / NBATCH);
  cudaMemset(p->d_flagSpec,0,4*NBATCH*NCHAN);
  float mn_bp[nBatches], tmp;

  //printf("fastflagger ");
  
  // loop over batches
  for (int batch = 0; batch < nBatches; batch++) {
  
    // load a batch
    //printf("transpose input %d of %d\n",batch+1,nBatches);
    begin = clock();
    transpose_input_handler(p->d_data+batch*NBATCH*NCHAN*p->NTIME,p->batch,p->NTIME,p->batch_stride);
    cudaDeviceSynchronize();
    end = clock();
    p->t7 += (float)(end - begin) / CLOCKS_PER_SEC;


    // loop over scrunches
    for (int scrnch=0;scrnch<p->nscrunches;scrnch++) {
      //printf("scrunch %d...",scrnch);
      if (scrnch==0)
	mn_bp[batch] = apply_scrunch(p, p->batch, p->mask, p->d_smooth, p->d_ts, p->NTIME, p->batch_stride, p->scrunches[scrnch].tscrunch,p->scrunches[scrnch].fscrunch, p->scrunches[scrnch].thresh,1,0,p->d_flagSpec,p->flag1,p->flag2);
      else
	tmp = apply_scrunch(p, p->batch, p->mask, p->d_smooth, p->d_ts, p->NTIME, p->batch_stride, p->scrunches[scrnch].tscrunch,p->scrunches[scrnch].fscrunch, p->scrunches[scrnch].thresh,1,0,p->d_flagSpec,p->flag1,p->flag2);
      cudaDeviceSynchronize();
    }
    tmp = apply_scrunch(p, p->batch, p->mask, p->d_smooth, p->d_ts, p->NTIME, p->batch_stride, 8, 8, 100., 0, 1, p->d_flagSpec,p->flag1,p->flag2);
    //    printf("\n");

    cudaDeviceSynchronize();

    // unload the batch
    begin = clock();
    transpose_output_handler(p->d_data+batch*NBATCH*NCHAN*p->NTIME,p->batch,p->NTIME,p->batch_stride);
    cudaMemcpy(p->h_flagSpec,p->d_flagSpec,4*NBATCH*NCHAN,cudaMemcpyDeviceToHost);
    end = clock();
    p->t8 += (float)(end - begin) / CLOCKS_PER_SEC;

    
    
    //printf("done\n");

    //printf("%g ",mn_bp);
    
  }
  //printf("\n");
  
  syslog(LOG_INFO,"fastflagger %g %g %g %g",mn_bp[0],mn_bp[1],mn_bp[2],mn_bp[3]);
  
}

// TEST cuda kernel to load test data
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void load_test(float * input, half * data, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] = __float2half(input[idx]);

}

// TEST cuda kernel to unload test data
// run with NBATCH*NCHAN*width/32 blocks of 32 threads 
__global__ void unload_test(float * output, half * data, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  output[idx] = __half2float(data[iidx]);

}



// input has shape [NBATCH, NCHAN, 15000]
void apply_batch_test(float * input, float * output, int width, int stride) {

  half * batch, * mask;
  //int stride = 16384;
  //int width = 14912;
  cudaMalloc(&batch, NBATCH * NCHAN * stride * sizeof(half));
  cudaMalloc(&mask, NBATCH * NCHAN * stride * sizeof(half));
  float * d_data;
  cudaMalloc(&d_data, NBATCH * NCHAN * width * sizeof(float));
    

  /// APPLY TEST HERE

  //transpose_input_handler(d_input,batch,width,stride);

  float * d_ts, begin, end;
  cudaMalloc(&d_ts, NBATCH * width * sizeof(float));
  cudaMemcpy(d_data,input,NBATCH * NCHAN * width * sizeof(float),cudaMemcpyHostToDevice);
  load_test<<<NBATCH*NCHAN*width/32,32>>>(d_data,batch,width,stride);

  begin = clock();
  //measure_ts<<<NBATCH*width,32>>>(batch, d_ts, width, stride);
  ts_correct(batch, d_ts, width, stride);
  cudaDeviceSynchronize();
  end = clock();
  printf("Time %g\n",(float)(end - begin) / CLOCKS_PER_SEC);
  
  cudaMemcpy(output,d_ts,NBATCH * width * sizeof(float),cudaMemcpyDeviceToHost);
  

  //transpose_output_handler(d_input,batch,width,stride);

  //unload_test<<<NBATCH*NCHAN*width/32,32>>>(d_data,batch,width,stride);
  //cudaMemcpy(output,d_data,sizeof(float)*NBATCH*NCHAN*width,cudaMemcpyDeviceToHost);
  //cudaMemcpy(output,d_input,NBATCH * NCHAN * width,cudaMemcpyDeviceToHost);
  
  
}

/************ END *************/


// function to do all dedispersion stuff
void dedisperse(pinfo *p, int beam) {

  cudaMemcpy(p->d_inputPacked,p->d_data+beam*NCHAN*p->NTIME,NCHAN*p->NTIME,cudaMemcpyDeviceToDevice);
  //cudaMemcpy(p->indata,p->d_data+beam*NCHAN*p->NTIME,NCHAN*p->NTIME,cudaMemcpyDeviceToHost);
  
  dedisp_error       derror;
  //const dedisp_byte* in = &((unsigned char *)(p->indata))[0];
  //dedisp_byte*       out = &((unsigned char *)(p->h_dedisp))[0];
  const dedisp_byte* in = (unsigned char *)(p->d_inputPacked);
  dedisp_byte*       out = (unsigned char *)(p->d_dedispPacked);
  dedisp_size        in_nbits = 8;
  dedisp_size        in_stride = NCHAN;// p->d_datapreT_step; //NCHAN * in_nbits/8;
  dedisp_size        out_nbits = 32;
  dedisp_size        out_stride = p->ntime_dedisp * out_nbits/8;
  unsigned           flags = 1 << 2;
  //unsigned           flags = 0;
  derror = dedisp_execute_adv(p->dedispersion_plan, p->NTIME,
                              in, in_nbits, in_stride,
                              out, out_nbits, out_stride,
                              flags);

  if (derror!=0) 
    std::cout << "DEDISP ERROR " << derror << std::endl;
  //cudaMemcpy2D(p->d_dedisp,p->d_dedisp_step,p->h_dedisp,4*p->ntime_dd,4*p->ntime_dd,p->ndms,cudaMemcpyHostToDevice);
  cudaMemcpy2D(p->d_dedisp,p->d_dedisp_step,p->d_dedispPacked,4*p->ntime_dedisp,4*p->ntime_dd,p->ndms,cudaMemcpyDeviceToDevice);
  
}

void smooth(pinfo *p, int scale) {

  NppiSize oSrcSize = {p->ntime_dd,p->ndms};
  NppiPoint oSrcOffset = {0,0};
  NppiSize oSizeROI = {p->ntime_dd,p->ndms};
  NppiSize oOutROI = {p->ntime_out,p->ndms-2};
  Npp32f * pKernel;
  Npp32f w[3] = {0.3,1.,0.3};
  Npp32f v, filtSum;
  NppiSize pKernelSize;
  NppiPoint oAnchor;

  /*int zeros_step;
  NppiSize oROI = {p->ntime_dd,1};
  Npp32f * zeros = nppiMalloc_32f_C1(p->ntime_dd,1,&(zeros_step));
  nppiSet_32f_C1R(0.,zeros,zeros_step,oROI);*/

  float * h_kernel;
  
  int smit = 0;
  for (int sm=(int)(p->minWidth); sm<(int)(p->maxWidth); sm *= 2) {
  
    // do smooth
    pKernelSize = {2*sm+1,3};
    cudaMalloc((void **)(&pKernel), 4*(2*sm+1)*3);
    h_kernel  = (float *)malloc(sizeof(float)*(2*sm+1)*3);
    filtSum = 0.;
    for (int i=0;i<3;i++) {
      for (int j=0;j<2*sm+1;j++) {
	v = 1.-0.5*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))+0.25*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))*0.25*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))-0.083*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))*0.083*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))*0.083*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355));
	h_kernel[i*(2*sm+1)+j] = w[i]*v*v;
	filtSum += w[i]*v*v;	
      }
    }
    oAnchor = {0,1};
    for (int i=0;i<(2*sm+1)*3;i++) h_kernel[i] /= filtSum;
    cudaMemcpy(pKernel,h_kernel,4*(2*sm+1)*3,cudaMemcpyHostToDevice);

    //nppiFilterBorder_32f_C1R(p->d_dedisp,p->d_dedisp_step,oSrcSize,oSrcOffset,p->boxes+smit*p->ndms*p->boxes_step/sizeof(float),p->boxes_step,oSizeROI,pKernel,pKernelSize,oAnchor,NPP_BORDER_REPLICATE);
    nppiFilterBorder_32f_C1R(p->d_dedisp,p->d_dedisp_step,oSrcSize,oSrcOffset,p->imbox,p->imbox_step,oSizeROI,pKernel,pKernelSize,oAnchor,NPP_BORDER_REPLICATE);

    // get rid of first and last DM, and maxWidth/2 from each edge
    cudaMemcpy2D(p->boxes+smit*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,p->imbox+p->imbox_step/sizeof(float)+(int)(p->maxWidth)/2,p->imbox_step,sizeof(float)*p->ntime_out,p->ndms-2,cudaMemcpyDeviceToDevice);
    
    smit++;
    cudaFree(pKernel);
    free(h_kernel);
    
  }

  // zero mean, std 1
  const Npp32f npm = -1.*p->mean;
  if (scale==1) {
    for (smit=0;smit<p->nboxcar;smit++) {

      nppiAddC_32f_C1IR(npm,p->boxes+smit*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,oOutROI);
      nppiMulC_32f_C1IR((const Npp32f)(1./p->stds[smit]),p->boxes+smit*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,oOutROI);

      // flag first DM trial
      //cudaMemcpy(p->boxes+smit*p->ndms*p->boxes_step/sizeof(float),zeros,p->ntime_dd*sizeof(float),cudaMemcpyDeviceToDevice);
      //cudaMemcpy(p->boxes+smit*p->ndms*p->boxes_step/sizeof(float)+(p->ndms-1)*p->ntime_dd,zeros,p->ntime_dd*sizeof(float),cudaMemcpyDeviceToDevice);

      
    }
  }

  //cudaFree(pKernel);
  //free(h_kernel);
  
}

// code to empirically measure thresholds
/*
// curand stuff
__global__ void setup_kernel(curandState* state, uint64_t seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}
__global__ void generate_randoms(curandState* globalState, float* randoms)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[tid];
    randoms[tid * 2 + 0] = curand_normal(&localState);
    randoms[tid * 2 + 1] = curand_normal(&localState);
}

void measure_thresholds(pinfo *p) {

  // generate data
  printf("THRESHOLD: Generating random values\n");
  int threads = 256;
  int blocks = (NCHAN/256)*p->NTIME / 2;
  int threadCount = blocks * threads;
  int N = blocks * threads * 2;
  curandState* dev_curand_states;
  float* randomValues;
  cudaMalloc(&dev_curand_states, threadCount * sizeof(curandState));
  cudaMalloc(&randomValues, N * sizeof(float));
  setup_kernel<<<blocks, threads>>>(dev_curand_states, time(NULL));
  generate_randoms<<<blocks, threads>>>(dev_curand_states, randomValues);  
  
  // prepare for dedispersion
  printf("THRESHOLD: dedisperse and smooth\n");
  NppiSize preROI = {NCHAN,p->NTIME};
  cudaMemcpy(p->dataFT,randomValues,4*N,cudaMemcpyDeviceToDevice);
  nppiScale_32f8u_C1R(p->dataFT,p->dataFT_step,p->d_datapreT,p->d_datapreT_step,preROI,-4.,10.);
  cudaMemcpy2D(p->data,NCHAN,p->d_datapreT,p->d_datapreT_step,NCHAN,p->NTIME,cudaMemcpyDeviceToHost);

  // dedisperse
  dedisperse(p);
  
  // smooth
  smooth(p,0);
  
  // measure stats
  printf("THRESHOLD: measure stats\n");
  Npp64f *pMean, *pStd;
  NppiSize oSizeROI = {p->ntime_dd,p->ndms};
  int nBufferSize;
  Npp8u * pDeviceBuffer;
  nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &nBufferSize);
  cudaMalloc((void **)(&pDeviceBuffer), nBufferSize);
  cudaMalloc((void **)(&pMean), p->nboxcar*8);
  cudaMalloc((void **)(&pStd), p->nboxcar*8);
  
  for (int i=0;i<p->nboxcar;i++) 
    nppiMean_StdDev_32f_C1R(p->boxes+i*p->ntime_dd*p->ndms,p->boxes_step,oSizeROI,pDeviceBuffer,pMean+i,pStd+i);

  double *hMean, *hStd;
  hMean = (double *)malloc(sizeof(double)*p->nboxcar);
  hStd = (double *)malloc(sizeof(double)*p->nboxcar);
  cudaMemcpy(hMean,pMean,sizeof(double)*p->nboxcar,cudaMemcpyDeviceToHost);
  cudaMemcpy(hStd,pStd,sizeof(double)*p->nboxcar,cudaMemcpyDeviceToHost);

  printf("(Boxcar) Mean Std\n");
  for (int i=0;i<p->nboxcar;i++)
    printf("(%d) %g %g\n",i,hMean[i],hStd[i]);

  // scale sigmas by 0.95 to accommodate reduction at increased DM due to more co-added data. 
  
  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStd);
  cudaFree(dev_curand_states);
  cudaFree(randomValues);
  free(hMean);
  free(hStd);
  
}
*/

// peak finding

void find_peaks(pinfo *p, int bm) {

  float * dmt_ptr = thrust::raw_pointer_cast(&p->dmt[0]);
  float * d_outputs = thrust::raw_pointer_cast(&p->output_values[0]);
  int * d_idxs = thrust::raw_pointer_cast(&p->output_indices[0]);
  int n_found;
  p->npeaks = 0;
  float myStd;

  for (int sm=0;sm<p->nboxcar;sm++) {

    // measure rms - should be 1
    //calculateStdDevFloat(float * d_data, int width, int height, int stride) {
    if (sm==0) {
      myStd = calculateStdDevFloat(p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),p->ntime_out,p->ndms-2,p->boxes_step/sizeof(float));
      if (bm==32) syslog(LOG_INFO,"STDDEV %g",myStd);
      if (myStd<1.2) myStd = 1.;
    }
    if (myStd<2.) {
    
      // copy to thrust vector
      //cudaMemcpy(dmt_ptr,p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),(p->ndms-2)*p->boxes_step,cudaMemcpyDeviceToDevice);
      cudaMemcpy2D(dmt_ptr,p->ntime_out*4,p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,p->ntime_out*4,p->ndms-2,cudaMemcpyDeviceToDevice);

      // Find indices and values of points greater than the threshold
      //  thrust::copy(p->dmt.begin(), p->dmt.begin() + 20, std::ostream_iterator<float>(std::cout, " "));
      thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::device,
								 thrust::make_counting_iterator(0),
								 thrust::make_counting_iterator(p->ntime_out*(p->ndms-2)),
								 p->dmt.begin(),
								 p->output_indices.begin(),
								 thrust::placeholders::_1 > p->snr*myStd);
      n_found = end-p->output_indices.begin();
      if (p->npeaks + n_found > MAX_GIANTS)
	n_found = MAX_GIANTS - p->npeaks;

      thrust::copy(thrust::make_permutation_iterator(p->dmt.begin(), p->output_indices.begin()),
		   thrust::make_permutation_iterator(p->dmt.end(), p->output_indices.begin()+n_found),
		   p->output_values.begin());
    
      // copy to host
      cudaMemcpy(p->peaks+p->npeaks, d_outputs, n_found*sizeof(float), cudaMemcpyDeviceToHost);
      thrust::for_each(p->output_indices.begin(), p->output_indices.begin()+n_found, thrust::placeholders::_1 += (p->ndms-2)*sm*p->ntime_out);
      cudaMemcpy(p->h_idxs+p->npeaks, d_idxs, n_found*sizeof(int), cudaMemcpyDeviceToHost);

      // iterate npeaks
      p->npeaks += n_found;

    }
    
  }
    
  // sort out stuff on host
  int tmp;
  int imax;
  //std::cout << p->npeaks << std::endl;
  if (p->out_npeaks+p->npeaks>MAX_GIANTS) 
    imax = MAX_GIANTS;
  else
    imax = p->out_npeaks+p->npeaks;
  for (int i=p->out_npeaks;i<imax;i++) {
    
    //printf("%d %d %d\n",p->h_idxs[i],(int)(p->h_idxs[i] % p->ntime_dd),(int)(p->h_idxs[i] / p->ntime_dd));
    p->out_peaks[i] = p->peaks[i-p->out_npeaks];
    p->out_beam[i] = bm;
    p->out_samp[i] = (int)(p->h_idxs[i-p->out_npeaks] % p->ntime_out);
    tmp = (int)(p->h_idxs[i-p->out_npeaks] / p->ntime_out);
    p->out_width[i] = (int)(tmp / (p->ndms-2));
    p->out_dm_idx[i] = (int)(tmp % (p->ndms-2)) + 1;

  }
  p->out_npeaks = imax;

}

// to clear all peaks
void clear_peaks(pinfo *p) {

  memset(p->out_peaks,0,MAX_GIANTS*4);
  memset(p->out_beam,0,MAX_GIANTS*4);
  memset(p->out_samp,0,MAX_GIANTS*4);
  memset(p->out_width,0,MAX_GIANTS*4);
  memset(p->out_dm_idx,0,MAX_GIANTS*4);
  p->out_npeaks = 0;
  

}

// output peaks
void output_peaks(pinfo *p, int samp) {

  // text output
  FILE *fout;
  fout=fopen(p->out_path,"a");
  
  if (p->out_format != 1) {

    for (int i=0;i<p->out_npeaks;i++) {
      /*if (p->samp[i]>p->maxWidth/2 && p->samp[i]<=p->ntime_dd-p->maxWidth/2)
      fprintf(fout,"A %g %d %g %d %d %g %d\n",p->peaks[i],p->samp[i]+samp,262.144e-6*(p->samp[i]+samp),p->width[i],p->dm_idx[i],p->DMs[p->dm_idx[i]],bm);
    else
    fprintf(fout,"B %g %d %g %d %d %g %d\n",p->peaks[i],p->samp[i]+samp,262.144e-6*(p->samp[i]+samp),p->width[i],p->dm_idx[i],p->DMs[p->dm_idx[i]],bm);*/
      fprintf(fout,"%g %d %d %g %d %d %g %d\n",p->out_peaks[i],p->out_samp[i]+samp,p->out_samp[i]+samp,262.144e-6*(p->out_samp[i]+samp)/86400.,p->out_width[i],p->out_dm_idx[i],p->DMs[p->out_dm_idx[i]],p->out_beam[i]+p->BEAM0);

    }
    fclose(fout);

  }

  // socket output
  std::ostringstream oss;
  
  if (p->out_format != 0) {
    
    try {
      ClientSocket client_socket ( p->coincidencer_host, p->coincidencer_port );

      oss << (int)(samp/p->gulp)+1 << std::endl;
      client_socket << oss.str();
      oss.flush();
      oss.str("");

      // record output
      for( int i=0; i<p->out_npeaks; i++ ) {
	oss << p->out_peaks[i] << " "
	    << p->out_samp[i]+samp << " "
	    << p->out_samp[i]+samp << " "
	    << 262.144e-6*(p->out_samp[i]+samp)/86400. << " "
	    << p->out_width[i] << " "
	    << p->out_dm_idx[i] << " "
	    << p->DMs[p->out_dm_idx[i]] << " "
	    << p->out_beam[i]+p->BEAM0 << std::endl;
	
	client_socket << oss.str();
	oss.flush();
	oss.str("");
      }
    }
      
    catch (SocketException& e )
      {
	std::cout << "SocketException was caught:" << e.description() << std::endl;
      }
  }

}

// deals with data IO
int main(int argc, char *argv[]) {
  /*  
  FILE *fin;
  fin = fopen(argv[1],"r");
  int width = atof(argv[2]);
  int stride = atof(argv[3]);
  float * input = (float *)malloc(sizeof(float)*NBATCH*NCHAN*width);
  float * output = (float *)malloc(sizeof(float)*NBATCH*width);

  int i=0;
  float t;
  while (!feof(fin)) {
    fscanf(fin,"%f\n",&t);
    input[i] = t;
    input[i+NCHAN*width] = input[i];
    i++;
  }
  fclose(fin);

  apply_batch_test(input,output,width,stride);

  fin=fopen("output.dat","w");
  for (int i=0;i<width*NBATCH;i++) {
    fprintf(fin,"%g\n",(float)(output[i]));
  }
  fclose(fin);
  

  free(input);
  free(output);
*/  
  

  
  // parse command line
  FILE *fconf;
  int core = -1;
  for (int i=1;i<argc;i++) {

    // configuration
    if (strcmp(argv[i],"-c")==0) {
      fconf=fopen(argv[i+1],"r");
      syslog(LOG_INFO,"Getting config from %s\n",argv[i+1]);
    }
    if (strcmp(argv[i],"-i")==0) {
      core = atoi(argv[i+1]);
    }
    // help
    if (strcmp(argv[i],"-h")==0) {
      help();
      exit(1);
    }

  }

  // set GPU ID
  cudaSetDevice(get_gpu_id(fconf));
  int currentDevice;
  cudaGetDevice(&currentDevice);

  // startup syslog message
  // using LOG_LOCAL0
  if (currentDevice==0)
    openlog ("dsaX_hella0", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  else
    openlog ("dsaX_hella1", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());


  syslog(LOG_INFO,"Using GPU ID %d\n",currentDevice);

  // Bind to cpu core
  if (core >= 0)
    {
      syslog(LOG_INFO,"binding to core %d\n", core);
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d\n", core);
    }
  
  
  // set up pipeline, allocate appropriate mem
  pinfo p;
  initialize(fconf,&p);  
  
  // begin read of data
  FILE *fin;
  float v;
  unsigned char * tmpbuf = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*p.gulp*NBEAMS*2);

  // DADA Header plus Data Unit 
  dada_hdu_t* hdu_in = 0;
  key_t in_key = DADA_BLOCK_KEY;
  char * header_in;
  uint64_t header_size = 0;
  uint64_t block_size;

  // dada input
  if (p.inp_format==0) {

    sscanf(p.inp_path, "%x", &in_key);
    hdu_in  = dada_hdu_create ();
    dada_hdu_set_key (hdu_in, in_key);
    dada_hdu_connect (hdu_in);
    dada_hdu_lock_read (hdu_in);
    header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
    ipcbuf_mark_cleared (hdu_in->header_block);
    block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
    syslog(LOG_INFO,"Connected to dada buffer\n");

  }
  
  // text file input
  if (p.inp_format==1) {
    fin=fopen(p.inp_path,"r");
    for (int i=0;i<p.NTIME*NCHAN;i++) {
      fscanf(fin,"%f\n",&v);
      p.data[i] = (unsigned char)(v);
    }
    fclose(fin);
  }

  // filterbank input
  if (p.inp_format==2) {
    fin=fopen(p.inp_path,"rb");
    
    int nbytes_header = read_header(fin);
    //    fclose(fin);
    //char * heade = (char *)malloc(sizeof(char)*nbytes_header);
    //(fin)=fopen(p.inp_path,"rb");
    //fread(heade, sizeof(char), nbytes_header, fin);
    //free(heade);
    syslog(LOG_INFO,"Finished with header (nbytes %d) of input filFile %s\n",nbytes_header,p.inp_path);
  }
    
  syslog(LOG_INFO,"Starting...\n");
  int samp = 0;
  if (p.inp_format!=1)
    samp = -(p.NTIME-p.gulp) + (int)(p.maxWidth)/2;
  int gulp = 0;
  
  //measure_thresholds(&p);

  // set up output
  FILE *fout, *fspec, *fbeam;
  int beamflags[NBEAMS], specflags[NCHAN];
  int bm;

  // loop over data gulps, figuring out at the end if we're finished

  // dada stuff
  char * block;
  uint64_t  bytes_read = 0;
  uint64_t block_id;

  // timer stuff
  float readt = 0., flagt = 0., dedispt = 0., smootht = 0., peakt = 0., outputt = 0.;
  float tot_time;
  clock_t begin, end;

  // outputs
  unsigned char * hodata = (unsigned char *)malloc(sizeof(unsigned char)*p.NTIME*NCHAN);
  //float * hodata = (float *)malloc(sizeof(float)*p.ntime_out*(p.ndms-2));
  FILE *ftest;
  int tot_flags = 0;
  
  while (finished==0) {

    // dada input
    if (p.inp_format==0) 
      block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    
    // set up logging and reset output
    //for (int i=0;i<NBEAMS;i++) beamflags[i] = 0;
    for (int i=0;i<NCHAN;i++) specflags[i] = 0;
    clear_peaks(&p);

    syslog(LOG_INFO,"Starting gulp %d\n",gulp);
    
    // loop over beams to read in data
    begin = clock();
    for (int bmm=0;bmm<NBEAMS;bmm++) {

      // get data from reader
      
      // dada input
      if (p.inp_format==0) {
	memcpy(p.data + bmm*p.NTIME*NCHAN + NCHAN*(p.NTIME-p.gulp),block+(bmm+p.BEAM_OFFSET)*p.gulp*NCHAN,p.gulp*NCHAN);
	memcpy(p.data + bmm*p.NTIME*NCHAN, p.rewinds + bmm*NCHAN*(p.NTIME-p.gulp), NCHAN*(p.NTIME-p.gulp));
	memcpy(p.rewinds + bmm*NCHAN*(p.NTIME-p.gulp), p.data + bmm*p.NTIME*NCHAN + NCHAN*p.gulp, NCHAN*(p.NTIME-p.gulp));
      }
      
      // filterbank input
      if (p.inp_format==2) {

	fread(tmpbuf, sizeof(unsigned char), 2*NBEAMS*p.gulp*NCHAN, fin);
	memcpy(p.data + bmm*p.NTIME*NCHAN + NCHAN*(p.NTIME-p.gulp),tmpbuf+(bmm+p.BEAM_OFFSET)*p.gulp*NCHAN,p.gulp*NCHAN);
	memcpy(p.data + bmm*p.NTIME*NCHAN, p.rewinds + bmm*NCHAN*(p.NTIME-p.gulp), NCHAN*(p.NTIME-p.gulp));
	memcpy(p.rewinds + bmm*NCHAN*(p.NTIME-p.gulp), p.data + bmm*p.NTIME*NCHAN + NCHAN*p.gulp, NCHAN*(p.NTIME-p.gulp));

	
      }

    }

    // copy to device
    cudaMemcpy(p.d_data,p.data,NBEAMS*p.NTIME*NCHAN,cudaMemcpyHostToDevice);
    end = clock();
    readt += (float)(end - begin) / CLOCKS_PER_SEC;

    if ((gulp>0 && p.inp_format!=1) || (p.inp_format==1)) {
      
      begin = clock();
      //printf("Flagging\n");
      fastflagger(&p);

      // write out to disk
      /*cudaMemcpy(hodata,p.d_data,NCHAN*p.NTIME,cudaMemcpyDeviceToHost);
      ftest = fopen("image.out","w");
      for (int i=0;i<NCHAN*p.NTIME;i++) 
	fprintf(ftest,"%f\n",(float)(hodata[i]));
	fclose(ftest);*/
      
      // deal with flags
      for (int j=0;j<NBATCH;j++) {
	for (int i=0;i<NCHAN;i++) {
	  //beamflags[bm] += (int)(p.h_flagSpec[j*NCHAN+i]);
	  specflags[i] += (int)(p.h_flagSpec[j*NCHAN+i]);
	  tot_flags += (int)(p.h_flagSpec[j*NCHAN+i]);
	}
      }
      end = clock();
      flagt += (float)(end - begin) / CLOCKS_PER_SEC;

      // loop over beams to dedisperse and search
      // check time, out_npeaks
      //printf("Looping over beams...\n");
      bm = 0;
      tot_time = readt+flagt;
      while ((bm<NBEAMS) && (tot_time<4.1) && (p.out_npeaks < MAX_GIANTS)) {
	//while ((bm<NBEAMS) && (p.out_npeaks < MAX_GIANTS)) {
      
	//printf("dedisperse\n");
	begin =	clock();
	dedisperse(&p,bm);
	end = clock();
        dedispt += (float)(end - begin) / CLOCKS_PER_SEC;

	// printf("begin smooth\n");
	begin = clock();
	smooth(&p,1);
	end = clock();
	smootht += (float)(end - begin) / CLOCKS_PER_SEC;

	
	//printf("rest\n");
	begin = clock();
	find_peaks(&p,bm);
	end = clock();
	peakt += (float)(end - begin) / CLOCKS_PER_SEC;
      

	tot_time = readt+flagt+dedispt+smootht+peakt;
	bm += 1;
	
      }

      begin = clock();
      // output peaks
      output_peaks(&p,samp);
      // output flags
      //fbeam = fopen(p.beamflags,"a");
      fspec = fopen(p.specflags,"a");
      //for (int i=0;i<NBEAMS;i++) fprintf(fbeam,"%d\n",beamflags[i]);
      for (int i=0;i<NCHAN;i++) fprintf(fspec,"%d\n",specflags[i]);
      //fclose(fbeam);
      fclose(fspec);
      end = clock();
      outputt += (float)(end - begin) / CLOCKS_PER_SEC;
      
    }

    // increment sample
    samp += p.gulp;
    gulp += 1;
    
    // assume only one gulp for text file input
    if (p.inp_format==1)
      finished = 1;

    // look for eof for fil input
    if (p.inp_format==2)
      if (feof(fin)) finished = 1;

    // close off dada block
    if (p.inp_format==0)
      ipcio_close_block_read (hdu_in->data_block, bytes_read);

    syslog(LOG_INFO,"Beamstats %d giants %d %d\n",bm,p.out_npeaks,tot_flags);
    tot_flags = 0;
    syslog(LOG_INFO,"processed %g s in read %g flag %g dedisp %g smooth %g peak %g output %g [%g]\n",(p.ntime_dd)*2.62144e-4,readt,flagt,dedispt,smootht,peakt,outputt,readt+flagt+dedispt+smootht+peakt+outputt);
    syslog(LOG_INFO,"Flagging: %g %g %g %g %g %g %g %g\n",p.t1,p.t2,p.t3,p.t4,p.t5,p.t6,p.t7,p.t8);
    readt = 0.;
    flagt = 0.;
    dedispt = 0.;
    smootht = 0.;
    peakt = 0.;
    outputt = 0.;
    p.t1 = 0.;
    p.t2 = 0.;
    p.t3 = 0.;
    p.t4 = 0.;
    p.t5 = 0.;
    p.t6 = 0.;
    p.t7 = 0.;
    p.t8 = 0.;
    
  }

  // deallocate stuff
  if (p.inp_format==0)
    dsaX_dbgpu_cleanup (hdu_in);
  deallocator(&p);  
  

}

