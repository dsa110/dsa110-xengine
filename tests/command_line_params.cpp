#include <command_line_params.h>

// General 
int core = 0;
bool debug = false;

// Data block HDU keys 
key_t in_key = REORDER_BLOCK_KEY;
key_t out_key = XGPU_BLOCK_KEY;

// Test mode
bool run_beamformer = false;
bool run_correlator = false;
double start_frequency = 1498.75;

// Test file
std::string test_filename;
int n_channels = 384;
int n_antennae = 63;
int n_pol = 2;
int n_times = 30720;

std::shared_ptr<dsaXApp> make_app(std::string app_description, std::string app_name) {

  auto dsaX_app = std::make_shared<dsaXApp>(app_description, app_name);
  dsaX_app->option_defaults()->always_capture_default();

  dsaX_app->add_option("--core", core, "Bind process to this CPU core [default 0]");
  dsaX_app->add_option("--debug", debug, "Send debug messages to syslog");
  dsaX_app->add_option("--in-key", in_key, "[default REORDER_BLOCK_KEY]");
  dsaX_app->add_option("--out-key", out_key, "[default XGPU_BLOCK_KEY]");
  dsaX_app->add_option("--run-beamformer", run_beamformer, "Run the beamformer [default false]");
  dsaX_app->add_option("--run-correlator", run_correlator, "Run the correlator [default false]");
  dsaX_app->add_option("--start-frequency", start_frequency, "start frequency (assumes 1498.75)");

  // Input file options
  dsaX_app->add_option("--test-filename", test_filename, "Name of file on which to run tests");
  dsaX_app->add_option("--n-channels", n_channels, "Number of frequency channels [default 384]");
  dsaX_app->add_option("--n-antennae", n_antennae, "Number of antennae [default 63]");
  dsaX_app->add_option("--n-pol", n_pol, "Number of polarizations [default 2]");
  dsaX_app->add_option("--n-times", n_times, "Number of times [default 30720]");

  return dsaX_app;
}
