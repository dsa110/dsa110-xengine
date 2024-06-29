#pragma once

#include <CLI.hpp>
#include <dsaX.h>

class dsaXApp : public CLI::App {
  
public:
  dsaXApp(std::string app_description = "", std::string app_name = "") : CLI::App(app_description, app_name) {};
  
  virtual ~dsaXApp() {};
};

std::shared_ptr<dsaXApp> make_app(std::string app_description = "dsaX internal test", std::string app_name = "");

// General 
extern int core;
extern bool debug;

// Data block HDU keys 
extern key_t in_key;
extern key_t out_key;

// Test mode
extern bool run_beamformer;
extern bool run_correlator;
extern double start_frequency;

// Test file
extern std::string input_filename;
extern std::string output_filename;
extern int n_channels;
extern int n_antennae;
extern int n_pol;
extern int n_times;
