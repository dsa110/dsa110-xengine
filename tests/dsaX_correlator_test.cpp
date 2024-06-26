#include <unistd.h> //DMH: replace with CLI
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <syslog.h>

// Include the dsaX.h header in your application
#include <dsaX.h>

using namespace std;

void usage() {
  fprintf (stdout,
	   "dsaX_beamformer_correlator [options]\n"
	   " -c if dsaX is CUDA enabled, use this GPU"
	   " -d send debug messages to syslog\n"
	   " -i in_key [default REORDER_BLOCK_KEY]\n"
	   " -o out_key [default XGPU_BLOCK_KEY]\n"
	   " -h print usage\n"
	   " -t binary file for test mode\n"
	   " -f flagants file\n"
	   " -a calib file\n"
	   " -s start frequency (assumes -0.244140625MHz BW)\n");
}

int main(int argc, char **argv) {

  // data block HDU keys
  key_t in_key = REORDER_BLOCK_KEY;
  key_t out_key = XGPU_BLOCK_KEY;
  
  // command line arguments
  int device_ordinal = 0;
  int arg = 0;
  int bf = 0;
  char ftest[200], fflagants[200], fcalib[200];
  float sfreq = 1498.75;
  
  while ((arg=getopt(argc,argv,"c:i:o:t:f:a:s:bdh")) != -1) {
    switch (arg) {
    case 'c':
      if (optarg) {
	device_ordinal = atoi(optarg);
	break;
      }
      else {
	syslog(LOG_ERR,"-c flag requires argument");
	usage();
	return EXIT_FAILURE;
      }
    case 'i':
      if (optarg) {
	if (sscanf (optarg, "%x", &in_key) != 1) {
	  syslog(LOG_ERR, "could not parse key from %s\n", optarg);
	  return EXIT_FAILURE;
	}
	break;
      } else {
	syslog(LOG_ERR,"-i flag requires argument");
	usage();
	return EXIT_FAILURE;
      }
    case 'o':
      if (optarg) {
	if (sscanf (optarg, "%x", &out_key) != 1) {
	  syslog(LOG_ERR, "could not parse key from %s\n", optarg);
	  return EXIT_FAILURE;
	}
	break;
      } else {
	syslog(LOG_ERR,"-o flag requires argument");
	usage();
	return EXIT_FAILURE;
      }
    case 't':
      if (optarg) {
	syslog(LOG_INFO, "test mode");
	if (sscanf (optarg, "%s", &ftest) != 1) {
	  syslog(LOG_ERR, "could not read test file name from %s\n", optarg);
	  return EXIT_FAILURE;
	}
	break;
      } else {
	syslog(LOG_ERR,"-t flag requires argument");
	usage();
	return EXIT_FAILURE;
      }
    case 'a':
      if (optarg) {
	syslog(LOG_INFO, "read calib file %s",optarg);
	if (sscanf (optarg, "%s", &fcalib) != 1) {
	  syslog(LOG_ERR, "could not read calib file name from %s\n", optarg);
	  return EXIT_FAILURE;
	}
	break;
      }
      else {
	syslog(LOG_ERR,"-a flag requires argument");
	usage();
	return EXIT_FAILURE;
      }
    case 'f':
      if (optarg) {
	syslog(LOG_INFO, "reading flag ants file %s",optarg);
	if (sscanf (optarg, "%s", &fflagants) != 1) {
	  syslog(LOG_ERR, "could not read flagants file name from %s\n", optarg);
	  return EXIT_FAILURE;
	}
	break;
      } else {
	syslog(LOG_ERR,"-f flag requires argument");
	usage();
	return EXIT_FAILURE;
      }
    case 's':
      if (optarg) {
	sfreq = atof(optarg);
	syslog(LOG_INFO, "start freq %g",sfreq);
	break;
      }
      else {
	syslog(LOG_ERR,"-s flag requires argument");
	usage();
	return EXIT_FAILURE;
      }
    case 'd':
      syslog (LOG_DEBUG, "Will excrete all debug messages");
      break;
    case 'h':
      usage();
      return EXIT_SUCCESS;
    }
  }
  
  std::cout << "NPACKETS_PER_BLOCK = " << NPACKETS_PER_BLOCK << std::endl;
  std::cout << "NCHAN = " << NCHAN << std::endl;
  std::cout << "NCHAN_PER_PACKET = " << NCHAN_PER_PACKET << std::endl;
  std::cout << "NPOL = " << NPOL << std::endl;
  std::cout << "NARM = " << 2 << std::endl;
  unsigned long long size = sizeof(char);
  size *= NPACKETS_PER_BLOCK;
  size *= NANTS;
  size *= NCHAN_PER_PACKET;
  size *= NPOL;
  size *= NCOMPLEX;
  std::cout << "(bytes) char size * NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*NPOL*NCOMPLEX = " << size << std::endl;
  std::cout << "Expected size of data array = " << (unsigned long long)(sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*NPOL*NCOMPLEX) << std::endl;
  std::cout << "Expected size of input array = " << (unsigned long long)(sizeof(char)*4*NANTS*NCHAN_PER_PACKET*NPOL*NCOMPLEX) << std::endl;
  
  //dsaX_init();  
  FILE *fin, *fout;
  std::cout << "Creating float output_array of size " << sizeof(float)*NBASE*NCHAN_PER_PACKET*2*2*4 << std::endl;
  uint64_t output_size = sizeof(float)*NBASE*NCHAN_PER_PACKET*2*2*4;
  std::cout << "Creating char input_array of size " << sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2 << std::endl;
  uint64_t input_size = sizeof(char)*NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2;

  float *output_data = (float *)malloc(output_size);
  char *input_data = (char *)malloc(input_size);
  
  // read one block of input data    
  for (int i=0;i<512;i++) {
    fin = fopen(ftest,"rb");
    fread(input_data + i*4*NANTS*NCHAN_PER_PACKET*2*2, 4*NANTS*NCHAN_PER_PACKET*2*2, 1, fin);
    fclose(fin);
  }

  // Peek at input data (delete after development is complete)
  //for (int i=0; i<10; i++) if(input_data[i] != 0) std::cout << "input[" << i <<"] = " << (float)input_data[i] << std::endl; 
  
  // run correlator and record output data
  syslog(LOG_INFO,"run correlator");
  dsaXCorrelator((void*)output_data, (void*)input_data);

  // Peek at output data (delete after development is complete)
  //for (int i=0; i<NBASE*NCHAN_PER_PACKET*2*2; i++) if(output_data[i] != 0) std::cout << "output " << i << " = " << output_data[i] << std::endl; 
  
  fout = fopen("output.dat","wb");
  fwrite((float *)output_data, sizeof(float), NBASE*NCHAN_PER_PACKET*2*2, fout);
  fclose(fout);
  
  // free
  free(input_data);
  free(output_data);
  //dsaX_end();
  
  return 0;
}

