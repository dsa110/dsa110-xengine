// -*- c++ -*-
/* assumes input and output block size is appropriate - will seg fault otherwise*/
/*
Workflow is similar for BF and corr applications
 - copy data to GPU, convert to half-precision and calibrate while reordering
 - do matrix operations to populate large output vector
 */

#include "dsaX_def.h"
#include "dsaX.h"
#include "dsaX_blas_interface.h"

//#include <cuda.h>
//#include "cuda_fp16.h"
//#include <cublas_v2.h>
//#include <cuda_runtime.h>

#include "dsaX_cuda_interface.h"

int DEBUG = 1;

void dsaX_dbgpu_cleanup(dada_hdu_t * in, dada_hdu_t * out)
{
  if (dada_hdu_unlock_read (in) < 0) syslog(LOG_ERR, "could not unlock read on hdu_in");
  dada_hdu_destroy (in);
  
  if (dada_hdu_unlock_write (out) < 0) syslog(LOG_ERR, "could not unlock write on hdu_out");
  dada_hdu_destroy (out);
  
} 

void usage() {
  fprintf (stdout,
	   "dsaX_beamformer_correlator [options]\n"
	   " -c core   bind process to CPU core [no default]\n"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default REORDER_BLOCK_KEY]\n"
	   " -o out_key [default XGPU_BLOCK_KEY]\n"
	   " -b run beamformer [default is to run correlator]\n"
	   " -h print usage\n"
	   " -t binary file for test mode\n"
	   " -f flagants file\n"
	   " -a calib file\n"
	   " -s start frequency (assumes -0.244140625MHz BW)\n");
}

// correlator function
// workflow: copy to device, reorder, stridedBatchedGemm, reorder
void dcorrelator(dmem *d) {

  // zero out output arrays
  cudaMemset(d->d_outr, 0, NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(half));
  cudaMemset(d->d_outi, 0, NCHAN_PER_PACKET*2*2*NANTS*NANTS*halfFac*sizeof(half));
  cudaMemset(d->d_output, 0, NCHAN_PER_PACKET*2*NANTS*NANTS*sizeof(float));
  
  // copy to device
  cudaMemcpy(d->d_input, d->h_input, NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2, cudaMemcpyHostToDevice);

  // reorder input
  reorder_input_device(d->d_input, d->d_tx, d->d_r, d->d_i);

  // ABSTRACT HERE START
  // not sure if essential
  cudaDeviceSynchronize();
  
  // set up for gemm
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  // gemm settings
  // input: [NCHAN_PER_PACKET, 2times, 2pol, NPACKETS_PER_BLOCK, NANTS]
  // output: [NCHAN_PER_PACKET, 2times, 2pol, NANTS, NANTS] 
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  const int m = NANTS;
  const int n = NANTS;
  const int k = NPACKETS_PER_BLOCK/halfFac;
  const half alpha = 1.;
  const half malpha = -1.;
  const int lda = m;
  const int ldb = n;
  const half beta0 = 0.;
  const half beta1 = 1.;
  const int ldc = m;
  const long long int strideA = NPACKETS_PER_BLOCK*NANTS/halfFac;
  const long long int strideB = NPACKETS_PER_BLOCK*NANTS/halfFac;
  const long long int strideC = NANTS*NANTS;
  const int batchCount = NCHAN_PER_PACKET*2*2*halfFac;

  // run strided batched gemm
  // ac
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,d->d_r,lda,strideA,
			    d->d_r,ldb,strideB,&beta0,
			    d->d_outr,ldc,strideC,
			    batchCount);
  // bd
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,d->d_i,lda,strideA,
			    d->d_i,ldb,strideB,&beta1,
			    d->d_outr,ldc,strideC,
			    batchCount);
  // -bc
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &malpha,d->d_i,lda,strideA,
			    d->d_r,ldb,strideB,&beta0,
			    d->d_outi,ldc,strideC,
			    batchCount);
  // ad
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			    &alpha,d->d_r,lda,strideA,
			    d->d_i,ldb,strideB,&beta1,
			    d->d_outi,ldc,strideC,
			    batchCount);

  // shown to be essential
  cudaDeviceSynchronize();

  // destroy stream
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  // ABSTRACT HERE END
  
  // reorder output data
  reorder_output_device(d);
}

/*
Beamformer:
 - initial data is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex] 
 - split into EW and NS antennas via cudaMemcpy: [NPACKETS_PER_BLOCK, NANTS/2, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
 - want [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, NANTS/2, 8chan, 2 times, 2 pol, 4-bit complex]
(single transpose operation)
 - weights are [NCHAN_PER_PACKET/8, NBEAMS, 4tim, NANTS/2, 8chan, 2 times, 2 pol] x 2
 - then fluff and run beamformer: output is [NCHAN_PER_PACKET/8, NBEAMS, NPACKETS_PER_BLOCK/4] (w column-major)
 - transpose and done! 

*/
// beamformer function
void dbeamformer(dmem * d) {

  // gemm settings - recall column major order assumed
  // stride over 48 chans
  cublasHandle_t cublasH = NULL;
  cublasCreate(&cublasH);
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = NPACKETS_PER_BLOCK/4;
  const int n = NBEAMS/2;
  const int k = 4*(NANTS/2)*8*2*2;
  const half alpha = 1.;
  const half malpha = -1.;
  const int lda = k;
  const int ldb = k;
  const half beta0 = 0.;
  const half beta1 = 1.;
  const int ldc = m;
  const long long int strideA = (NPACKETS_PER_BLOCK)*(NANTS/2)*8*2*2;
  const long long int strideB = (NBEAMS/2)*4*(NANTS/2)*8*2*2;
  const long long int strideC = (NPACKETS_PER_BLOCK/4)*NBEAMS/2;
  const int batchCount = NCHAN_PER_PACKET/8;
  long long int i1, i2;//, o1;
  
  // create streams
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // timing
  // copy, prepare, cublas, output
  clock_t begin, end;

  // do big memcpy
  begin = clock();
  cudaMemcpy(d->d_big_input,d->h_input,NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*4,cudaMemcpyHostToDevice);
  end = clock();
  d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
  
  // loop over halves of the array
  for (int iArm=0;iArm<2;iArm++) {
  
    // zero out output arrays
    cudaMemset(d->d_bigbeam_r,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(half));
    cudaMemset(d->d_bigbeam_i,0,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2)*sizeof(half));
    cudaDeviceSynchronize();
    
    // copy data to device
    // initial data: [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
    // final data: need to split by NANTS.
    begin = clock();
    for (i1=0;i1<NPACKETS_PER_BLOCK;i1++) 
      cudaMemcpy(d->d_input+i1*(NANTS/2)*NCHAN_PER_PACKET*4,d->d_big_input+i1*(NANTS)*NCHAN_PER_PACKET*4+iArm*(NANTS/2)*NCHAN_PER_PACKET*4,(NANTS/2)*NCHAN_PER_PACKET*4,cudaMemcpyDeviceToDevice);
    end = clock();
    d->cp += (float)(end - begin) / CLOCKS_PER_SEC;
    
    // do reorder and fluff of data to real and imag
    begin = clock();
    dim3 dimBlock1(16, 8), dimGrid1(NCHAN_PER_PACKET/8/16, (NPACKETS_PER_BLOCK)*(NANTS/2)/16);
    transpose_input_bf<<<dimGrid1,dimBlock1>>>((double *)(d->d_input),(double *)(d->d_tx));
    fluff_input_bf<<<NPACKETS_PER_BLOCK*(NANTS/2)*NCHAN_PER_PACKET*2*2/128,128>>>(d->d_tx,d->d_br,d->d_bi);
    end = clock();
    d->prep += (float)(end - begin) / CLOCKS_PER_SEC;

    // large matrix multiply to get real and imag outputs
    // set up for gemm
    cublasSetStream(cublasH, stream);
    i2 = iArm*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8); // weights offset
          
    // run strided batched gemm
    begin = clock();
    // ac
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_br,lda,strideA,
			      d->weights_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_r,ldc,strideC,
			      batchCount);
    // -bd
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &malpha,d->d_bi,lda,strideA,
			      d->weights_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_r,ldc,strideC,
			      batchCount);
    // bc
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_bi,lda,strideA,
			      d->weights_r+i2,ldb,strideB,&beta0,
			      d->d_bigbeam_i,ldc,strideC,
			      batchCount);
    // ad
    cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
			      &alpha,d->d_br,lda,strideA,
			      d->weights_i+i2,ldb,strideB,&beta1,
			      d->d_bigbeam_i,ldc,strideC,
			      batchCount);
      
    cudaDeviceSynchronize();
    end = clock();
    d->cubl += (float)(end - begin) / CLOCKS_PER_SEC;
      
        
    // simple formation of total power and scaling to 8-bit in transpose kernel
    begin = clock();
    dim3 dimBlock(16, 8), dimGrid((NBEAMS/2)*(NPACKETS_PER_BLOCK/4)/16, (NCHAN_PER_PACKET/8)/16);
    transpose_scale_bf<<<dimGrid,dimBlock>>>(d->d_bigbeam_r,d->d_bigbeam_i,d->d_bigpower+iArm*(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*(NBEAMS/2));
    end = clock();
    d->outp += (float)(end - begin) / CLOCKS_PER_SEC;
  }

  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);

  // form sum over times
  //sum_beam<<<24576,512>>>(d->d_bigpower,d->d_chscf);
  
}


// MAIN
#if 0
int main (int argc, char *argv[]) {

  cudaSetDevice(0);
  
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_bfCorr", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;

  // data block HDU keys
  key_t in_key = REORDER_BLOCK_KEY;
  key_t out_key = XGPU_BLOCK_KEY;
  
  // command line arguments
  int core = -1;
  int arg = 0;
  int bf = 0;
  int test = 0;
  char ftest[200], fflagants[200], fcalib[200];
  float sfreq = 1498.75;
  
  while ((arg=getopt(argc,argv,"c:i:o:t:f:a:s:bdh")) != -1)
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
	      test = 1;
	      syslog(LOG_INFO, "test mode");
	      if (sscanf (optarg, "%s", &ftest) != 1) {
		syslog(LOG_ERR, "could not read test file name from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-t flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'a':
	  if (optarg)
            {
	      syslog(LOG_INFO, "read calib file %s",optarg);
	      if (sscanf (optarg, "%s", &fcalib) != 1) {
		syslog(LOG_ERR, "could not read calib file name from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-a flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'f':
	  if (optarg)
            {
	      syslog(LOG_INFO, "reading flag ants file %s",optarg);
	      if (sscanf (optarg, "%s", &fflagants) != 1) {
		syslog(LOG_ERR, "could not read flagants file name from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 's':
	  if (optarg)
            {
	      sfreq = atof(optarg);
	      syslog(LOG_INFO, "start freq %g",sfreq);
 	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-s flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'd':
	  //DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'b':
	  bf=1;
	  syslog (LOG_NOTICE, "Running beamformer, NOT correlator");
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

  // allocate device memory
  dmem d;
  initialize_device_memory(&d,bf);

  // set up for beamformer
  FILE *ff;
  int iii;
  if (bf) {

    if (!(ff=fopen(fflagants,"r"))) {
      syslog(LOG_ERR,"could not open flagants file\n");
      exit(1);
    }
    d.nflags=0;
    while (!feof(ff)) {
      fscanf(ff,"%d\n",&d.flagants[iii]);
      d.nflags++;
    }
    fclose(ff);

    if (!(ff=fopen(fcalib,"rb"))) {
      syslog(LOG_ERR,"could not open calibss file\n");
      exit(1);
    }
    fread(d.h_winp,NANTS*2+NANTS*(NCHAN_PER_PACKET/8)*2*2,4,ff);
    fclose(ff);

    for (iii=0;iii<(NCHAN_PER_PACKET/8);iii++)
      d.h_freqs[iii] = 1e6*(sfreq-iii*250./1024.);
    cudaMemcpy(d.d_freqs,d.h_freqs,sizeof(float)*(NCHAN_PER_PACKET/8),cudaMemcpyHostToDevice);

    // calculate weights
    calc_weights(&d);
    
  }

  // test mode
  FILE *fin, *fout;
  uint64_t output_size;
  char * output_data;//, * o1;
  if (test) {

    // read one block of input data    
    d.h_input = (char *)malloc(sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2);
    for (int i=0;i<512;i++) {
      fin = fopen(ftest,"rb");
      fread(d.h_input+i*4*NANTS*NCHAN_PER_PACKET*2*2,4*NANTS*NCHAN_PER_PACKET*2*2,1,fin);
      fclose(fin);
    }

    // run correlator or beamformer, and output data
    if (bf==0) {
      if (DEBUG) syslog(LOG_INFO,"run correlator");
      dcorrelator(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      output_size = NBASE*NCHAN_PER_PACKET*2*2*4;
      output_data = (char *)malloc(output_size);
      cudaMemcpy(output_data,d.d_output,output_size,cudaMemcpyDeviceToHost);

      fout = fopen("output.dat","wb");
      fwrite((float *)output_data,sizeof(float),NBASE*NCHAN_PER_PACKET*2*2,fout);
      fclose(fout);
    }
    else {
      if (DEBUG) syslog(LOG_INFO,"run beamformer");
      dbeamformer(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      output_size = (NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*NBEAMS;
      output_data = (char *)malloc(output_size);
      cudaMemcpy(output_data,d.d_bigpower,output_size,cudaMemcpyDeviceToHost);

      /*output_size = 2*2*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8);
      o1 = (char *)malloc(output_size);
      cudaMemcpy(o1,d.weights_r,output_size,cudaMemcpyDeviceToHost);*/
	
      

      fout = fopen("output.dat","wb");
      fwrite((unsigned char *)output_data,sizeof(unsigned char),output_size,fout);
      //fwrite(o1,1,output_size,fout);
      fclose(fout);
    }

	
    // free
    free(d.h_input);
    free(output_data);
    //free(o1);
    deallocate_device_memory(&d,bf);

    exit(1);
  }
  


  
  // DADA stuff
  
  syslog (LOG_INFO, "creating in and out hdus");
  
  hdu_in  = dada_hdu_create (0);
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to dada buffer in");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to dada buffer in");
    return EXIT_FAILURE;
  }
  
  hdu_out  = dada_hdu_create (0);
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

  syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");  
  
  // get block sizes and allocate memory
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  syslog(LOG_INFO, "main: have input and output block sizes %lu %lu\n",block_size,block_out);
  if (bf==0) 
    syslog(LOG_INFO, "main: EXPECT input and output block sizes %d %d\n",NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2,NBASE*NCHAN_PER_PACKET*2*2*4);
  else
    syslog(LOG_INFO, "main: EXPECT input and output block sizes %d %d\n",NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2,(NPACKETS_PER_BLOCK/4)*(NCHAN_PER_PACKET/8)*NBEAMS);
  uint64_t  bytes_read = 0;
  //char * block;
  char * output_buffer;
  output_buffer = (char *)malloc(block_out);
  uint64_t written, block_id;
  
  // get things started
  bool observation_complete=0;
  //bool started = 0;
  syslog(LOG_INFO, "starting observation");
  int blocks = 0;
  //clock_t begin, end;
  //double time_spent;
  
  while (!observation_complete) {

    if (DEBUG) syslog(LOG_INFO,"reading block");    
    
    // open block
    d.h_input = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);

    // do stuff
    //begin = clock();
    // loop
    if (bf==0) {
      if (DEBUG) syslog(LOG_INFO,"run correlator");
      dcorrelator(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      cudaMemcpy(output_buffer,d.d_output,block_out,cudaMemcpyDeviceToHost);
    }
    else {
      if (DEBUG) syslog(LOG_INFO,"run beamformer");
      dbeamformer(&d);
      if (DEBUG) syslog(LOG_INFO,"copy to host");
      cudaMemcpy(output_buffer,d.d_bigpower,block_out,cudaMemcpyDeviceToHost);
    }
    //end = clock();
    //time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << "spent time " << d.cp << " " << d.prep << " " << d.cubl << " " << d.outp << " s" << endl;
    
    // write to output

    // write to host
    written = ipcio_write (hdu_out->data_block, (char *)(output_buffer), block_out);
    if (written < block_out)
      {
	syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	return EXIT_FAILURE;
      }
    
    if (DEBUG) syslog(LOG_INFO, "written block %d",blocks);	    
    blocks++;
    // loop end
    
      
    // finish up
    if (bytes_read < block_size)
      observation_complete = 1;
    
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
    
  }

  // finish up
  free(output_buffer);
  deallocate_device_memory(&d,bf);
  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  
}
#endif

