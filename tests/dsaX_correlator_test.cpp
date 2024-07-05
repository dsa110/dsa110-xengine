#include <unistd.h> //DMH: replace with CLI
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <syslog.h>
#include <random>

// Include this file to access input parameters
#include "command_line_params.h"

// Include the dsaX.h header in your application
#include <dsaX.h>

using namespace std;

// The class offers entire file content read/write in single operation
class BinaryFileVector : public std::vector<char>
{
public:

  using std::vector<char>::vector;

  bool loadFromFile(const char *fileName) noexcept
  {
    // Try to open a file specified by its name    
    std::ifstream file(fileName, std::ios::in | std::ios::binary);
    if (!file.is_open() || file.bad())
      return false;

    // Clear whitespace removal flag
    file.unsetf(std::ios::skipws);

    // Determine size of the file
    file.seekg(0, std::ios_base::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios_base::beg);

    // Discard previous vector content
    resize(0);
    reserve(0);
    shrink_to_fit();

    // Order to prealocate memory to avoid unnecessary reallocations due to vector growth
    reserve(fileSize);

    // Read entire file content into prealocated vector memory
    insert(begin(),
	   std::istream_iterator<char>(file),
	   std::istream_iterator<char>());

    // Make sure entire content is loaded
    if(size() == fileSize) {
      std::cout << "Successfully read file of size " << fileSize << std::endl;
      return true;
    } else {
      std::cout << "Unexpected file size." << std::endl;
      return false;
    }
  }

  bool saveToFile(const char *fileName) const noexcept
  {
    // Write entire vector content into a file specified by its name
    std::ofstream file(fileName, std::ios::out | std::ios::binary);
    try {
      file.write((const char *) data(), size());
    }
    catch (...) {
      return false;
    }

    // Determine number of bytes successfully stored in file
    size_t fileSize = file.tellp();
    if(size() == fileSize) {
      std::cout << "Successfully wrote file of size " << fileSize  << std::endl;
      return true;
    } else {
      std::cout << "Unexpected file size." << std::endl;
      return false;
    }
  }
};

int main(int argc, char **argv) {

  // Parse command line
  auto app = make_app();  
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  
  int device_ordinal = 0;
  int packet_size = 4608;

  // Create a data array for a single call to the correlator class
  FILE *fin, *fout;
  uint64_t sz, in_block_size, rd_size;
  in_block_size = NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2;
  
  std::cout << "Creating char file_array of size " << (1.0*sizeof(char)*in_block_size)/pow(1024,2) << " MB." << std::endl;
  char *file_data = (char *)malloc(in_block_size);  

  // read one block of input data  
  // get size of file
  if(!input_rands) {
    std::cout << "attempting to read file " << input_filename.c_str() << std::endl; 
    fin = fopen(input_filename.c_str(), "rb");
    fseek(fin, 0L, SEEK_END);
    sz = ftell(fin);
    if(sz != packet_size) {
      cout << "Error: packet size " << packet_size << " and file size " << sz << " are unequal." << endl;
      exit(0);
    }
    rewind(fin);

    // figure out how many reps and chunks to read with
    int nreps, nchunks;
    if (sz > in_block_size) {
      nreps = (int)(sz/in_block_size);
      rd_size = in_block_size;
    }
    else {
      nchunks = (int)(in_block_size/sz);
      rd_size = sz;
    }

    cout << "Packet size = " << sz << endl;
    cout << "rd size = " << rd_size << endl;
    for (int reps = 0; reps<nreps; reps++) {
      for (int chunks = 0; chunks < nchunks; chunks++) {	
	fread(file_data + (chunks + reps * nchunks)*rd_size , rd_size, 1, fin);
      }
    }
  } else {
    int n_rand = in_block_size/sizeof(uint64_t);
    uint64_t *input_rand = (uint64_t*)malloc(n_rand);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    gen.seed(1234);
    std::uniform_int_distribution<uint64_t> dis;
    for (int i = 0; i < n_rand; i++) input_rand[i] = dis(gen);
    //for (int i = 0; i < n_rand; i++) input_rand[i] = (uint64_t)1234;
    memcpy(file_data, (void*)input_rand, n_rand);
    free(input_rand);
  }
  
  // Start dsaX program
  //---------------------------------------
  timer::Timer<std::chrono::microseconds, std::chrono::high_resolution_clock> test_timer;

  dsaXInit(device_ordinal);
  
  // Create Correlator class instance.
  dsaXCorrParam param = newDsaXCorrParam();
  param.blas_lib = DSA_BLAS_LIB_CUBLAS;
  param.data_type = DSA_BLAS_DATATYPE_4b_COMPLEX;
  param.data_order = DSA_BLAS_DATAORDER_ROW;
  param.n_streams = n_streams;
  printDsaXCorrParam(param);
  
  auto correlator = new Correlator(&param);

  // Create GPU registered memory if using CUDA 
  uint64_t input_size = n_streams*sizeof(char)*in_block_size;
  std::cout << "Creating char input array of size " << input_size << " bytes." << std::endl;
  void *input_data = dsaXHostRegister(input_size);
  // Populate with random data. Each stream has the same data
  // To ensure the concurrency does not pollute accross streams. 
  for (int i = 0; i<n_streams; i++) memcpy((char*)input_data + i * in_block_size, file_data, in_block_size);

  // Create GPU registered output array
  uint64_t output_size = n_streams * sizeof(float) * NBASE*NCHAN_PER_PACKET*2*2;
  std::cout << "Creating float output_array of size " << output_size << " bytes." << std::endl;
  void *output_data = dsaXHostRegister(output_size);

  // Ensure test output array is zero
  memset(output_data, 0, output_size);
  
  std::cout << "Total input size = " << (1.0 * input_size)/pow(1024,3) << " GB." << endl;
  std::cout << "Expected output size = " << (1.0 * output_size)/pow(1024,3) << " GB." << endl;
  
  test_timer.start();  
  correlator->compute(output_data, input_data);
  test_timer.stop();
  
  //std::cout << "Output peek " << std::endl;
  float *p = (float*)output_data;
  for(int i=0; i<8; i++) cout << "output[" << i << "] = " << p[i] << endl;
  
  if(write_output) {
    fout = fopen(output_filename.c_str(),"ab");
    fwrite((unsigned char *)output_data, sizeof(unsigned char *), sizeof(float)*output_size, fout);
    fclose(fout);
  }
  
  delete correlator;
  dsaXEnd();

  std::cout << "Test time = " << (1.0*test_timer.elapsed().count())/(1e6) << " seconds. " << endl;
  
  // End dsaX program
  //---------------------------------------

  // free local data
  free(input_data);
  free(output_data);
  return 0;
  
  /*  
  // Read data
  BinaryFileVector binaryFileVector;

  
  if (!binaryFileVector.loadFromFile(test_filename.c_str())) {
    std::cout << "Failed to read the file." << std::endl;
    return 0;
  }
  
  // read one block of input data
  for (int i=0;i<512;i++) {
    //fin = fopen(test_filename,"rb");
    //fread(input_data + i*4*NANTS*NCHAN_PER_PACKET*2*2, 4*NANTS*NCHAN_PER_PACKET*2*2, 1, fin);
    //fclose(fin);
  }

  for (int i=0;i<512;i++) {
    memcpy(input_data + i*binaryFileVector.size(), binaryFileVector.data(), binaryFileVector.size());
  }
  
  // Peek at input data (delete after development is complete)
  for (int i=0; i<8; i++) inspectPackedData(input_data[i], i);  

  // Peek at output data (delete after development is complete)
  for (int i=0; i<NBASE*NCHAN_PER_PACKET*2*2; i++) if(output_data[i] != 0) std::cout << "output " << i << " = " << output_data[i] << std::endl;
  //for (int i=0; i<8; i++) std::cout << "output " << i << " = " << output_data[i] << std::endl; 

  if (!binaryFileVector.saveToFile("output.dat")) {
    std::cout << "Failed to write a file." << std::endl;
    return 0;
  } else {
    std::cout << "Successfully wrote file." << std::endl;
  }
  
  
  fout = fopen("output.dat","wb");
  fwrite((float *)output_data, sizeof(float), NBASE*NCHAN_PER_PACKET*2*2, fout);
  fclose(fout);
  */
      

}
