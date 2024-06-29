#include <unistd.h> //DMH: replace with CLI
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <syslog.h>

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
  FILE *fin, *fout;
  uint64_t sz, output_size, in_block_size, rd_size;
  in_block_size = NPACKETS_PER_BLOCK*NANTS*NCHAN_PER_PACKET*2*2;
  int nreps = 1, nchunks = 1;

  // read one block of input data  
  // get size of file
  std::cout << "attempting to read file " << input_filename.c_str() << std::endl; 
  fin=fopen(input_filename.c_str(), "rb");
  fseek(fin, 0L, SEEK_END);
  sz = ftell(fin);
  rewind(fin);

  // figure out how many reps and chunks to read with
  if (sz > in_block_size) {
    nreps = (int)(sz/in_block_size);
    rd_size = in_block_size;
  }
  else {
    nchunks = (int)(in_block_size/sz);
    rd_size = sz;
  }

  // Start dsaX program
  //---------------------------------------
  dsaXInit(device_ordinal);
  
  // Create Correlator class instance.
  dsaXCorrParam param = newDsaXCorrParam();
  param.blas_lib = DSA_BLAS_LIB_CUBLAS;
  param.data_type = DSA_BLAS_DATATYPE_4b_COMPLEX;
  param.data_order = DSA_BLAS_DATAORDER_ROW;
  printDsaXCorrParam(param);
  auto correlator = new Correlator(&param);

  output_size = NBASE*NCHAN_PER_PACKET*2*2*4;
  std::cout << "Creating char output_array of size " << (1.0*sizeof(float)*NBASE*NCHAN_PER_PACKET*2*2)/pow(1024,2) << " MB." << std::endl;
  char *output_data = (char *)malloc(output_size);

  std::cout << "Creating char input_array of size " << (1.0*sizeof(char)*in_block_size)/pow(1024,2) << " MB." << std::endl;
  char *input_data = (char *)malloc(in_block_size);
  
  std::cout << "Computing " << nreps << " repetitions of " << nchunks << " chunks of input data of size " << rd_size << " bytes." << endl;
  std::cout << "Total input size = " << (1.0 * nreps * nchunks * rd_size)/pow(1024,3) << " GB." << endl;
  std::cout << "Expected output size = " << (1.0 * nreps * nchunks * output_size)/pow(1024,3) << " GB." << endl;
  
  // Loop over reps and chunks
  for (int reps = 0; reps<nreps; reps++) {
    for (int chunks = 0; chunks<nchunks; chunks++) {

      // Read input file
      if (chunks>0) rewind(fin);
      fread(input_data + chunks*rd_size, rd_size, 1, fin);

      //std::cout << "Input peek " << std::endl;
      //for (int i=0; i<8; i++) inspectPackedData(input_data[i], i);
      
      // run correlator and record output data
      //dsaXCorrelator((void*)output_data, (void*)input_data, &param);
      correlator->compute((void*)output_data, (void*)input_data);
      
      //std::cout << "Output peek " << std::endl;
      //for(int i=0; i<output_size; i++) inspectPackedData(output_data[i], i, true);

      fout = fopen(output_filename.c_str(),"ab");
      fwrite((unsigned char *)output_data, sizeof(unsigned char *), sizeof(float)*output_size, fout);
      fclose(fout);
      exit(0);
    }
  }

  dsaXEnd();
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
