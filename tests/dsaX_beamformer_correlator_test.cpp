#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Include the dsaX.h header in your application
//#include <dsaX.h>

int main(int argc, char **argv) {

  /*
  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_bfCorr", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  // DADA Header plus Data Unit 
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
  if (core >= 0) {
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

      // output_size = 2*2*4*(NANTS/2)*8*2*2*(NBEAMS/2)*(NCHAN_PER_PACKET/8);
      // o1 = (char *)malloc(output_size);
      // cudaMemcpy(o1,d.weights_r,output_size,cudaMemcpyDeviceToHost);
	
      

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
  
  return 0;
  */
}
