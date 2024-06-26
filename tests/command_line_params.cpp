#include <command_line_params.h>

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

