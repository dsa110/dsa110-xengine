#include <unistd.h> //DMH: replace with CLI
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <syslog.h>
#include <random>

using namespace std;

// Include this file to access input parameters
#include "command_line_params.h"

// Include this file to access test utilities
/**
 * Promote complex char riri... data to planar half rr.. ii.. 
 *
 * @param[out] inr float precision real array
 * @param[out] ini float precision imag array
 * @param[in]  input char precision complex array
 * @param[in]  rows number of rows
 * @param[in]  cols number of cols
 */
template <typename prec> void promoteComplexCharToFloat(prec *output, const char *input, const int rows, const int cols) {
  
#pragma omp parallel for collapse(2)
  int idx = 0;
  for(int i=0; i<rows; i++) {
    for(int j=0; j<cols; j++) {
      idx = i * cols + j;
      
      // 15 in unsigned char binary is 00001111. Perform bitwise & on 15 and input char data iiiirrrr
      // to get real part 4 bit data.
      // 0000rrrr
      // Bit shift this result by 4 to the left.
      // rrrr0000
      // Cast to signed char.
      // +-rrr0000
      // Bitshift mantisa only to the right by 4 bits
      // +-0000rrr
      // Cast to float and use CUDA intrinsic to cast to signed half
      output[2*idx] = (prec)((char)((   (unsigned char)(input[idx]) & (unsigned char)(15)  ) << 4) >> 4);
      
      // 240 in unsigned char binary is 11110000. Perform bitwise & on 240 and input char data iiiirrrr
      // to get imag part 4 bit data
      // iiii0000.
      // Cast to signed char
      // +-iii0000
      // Bitshift mantisa only to the right by 4 bits
      // +-0000iii
      // Cast to float and use CUDA intrinsic to cast to signed half
      output[2*idx+1] = (prec)((char)((   (unsigned char)(input[idx]) & (unsigned char)(240)  )) >> 4);
    }
  }
}

// Assume ROW ordered data in interleaved format
template <typename prec> void host_MdagM_gemm(const prec *A, const prec *B, prec *C, const int m, const int n, const int k) {
  
#pragma omp parallel for collapse(2)
  for(int i=0; i<m; i++) {
    for(int j=0; j<n; j++) {
      
      // Get C index
      int C_idx_r = 2*(i * n + j);
      int C_idx_i = 2*(i * n + j) + 1;
      C[C_idx_r] = 0.0;
      C[C_idx_i] = 0.0;
      for(int l=0; l<k; l++) {

	// A is conjugated
	int A_idx_r = 2*(l * m + i);
	int A_idx_i = 2*(l * m + i) + 1;
	
	int B_idx_r = 2*(l * n + j);
	int B_idx_i = 2*(l * n + j) + 1;

	// Compute Adag * B = C
	C[C_idx_r] += A[A_idx_r] * B[B_idx_r] + A[A_idx_i] * B[B_idx_i];
	C[C_idx_i] += A[A_idx_r] * B[B_idx_i] - A[A_idx_i] * B[B_idx_r];
      }
    }
  }
}

// Assume ROW ordered data in interleaved format
template <typename prec> prec test_hermiticity(const prec *C, const int m, const int n) {

  prec frob_norm = 0.0;
  
#pragma omp parallel for collapse(2) reduction (+:frob_norm)
  for(int i=0; i<m; i++) {
    for(int j=0; j<n; j++) {

      // Get Cdag index
      int Cd_idx_r = 2*(j * m + i);
      int Cd_idx_i = 2*(j * m + i) + 1;
      
      // Get C index
      int C_idx_r = 2*(i * n + j);
      int C_idx_i = 2*(i * n + j) + 1;

      double diff = pow((C[C_idx_r] - C[Cd_idx_r]), 2);
      diff       += pow((C[C_idx_i] + C[Cd_idx_i]), 2);
      frob_norm = frob_norm + diff;
    }
  }
  return frob_norm/(m*n*2);
}

// Include the dsaX.h header in your application
#include <dsaX.h>

// The class offers entire file content read/write in single operation
class BinaryFileVector : public vector<char>
{
public:

  using vector<char>::vector;

  bool loadFromFile(const char *fileName) noexcept
  {
    // Try to open a file specified by its name    
    ifstream file(fileName, ios::in | ios::binary);
    if (!file.is_open() || file.bad())
      return false;

    // Clear whitespace removal flag
    file.unsetf(ios::skipws);

    // Determine size of the file
    file.seekg(0, ios_base::end);
    size_t fileSize = file.tellg();
    file.seekg(0, ios_base::beg);

    // Discard previous vector content
    resize(0);
    reserve(0);
    shrink_to_fit();

    // Order to prealocate memory to avoid unnecessary reallocations due to vector growth
    reserve(fileSize);

    // Read entire file content into prealocated vector memory
    insert(begin(),
	   istream_iterator<char>(file),
	   istream_iterator<char>());

    // Make sure entire content is loaded
    if(size() == fileSize) {
      cout << "Successfully read file of size " << fileSize << endl;
      return true;
    } else {
      cout << "Unexpected file size." << endl;
      return false;
    }
  }

  bool saveToFile(const char *fileName) const noexcept
  {
    // Write entire vector content into a file specified by its name
    ofstream file(fileName, ios::out | ios::binary);
    try {
      file.write((const char *) data(), size());
    }
    catch (...) {
      return false;
    }

    // Determine number of bytes successfully stored in file
    size_t fileSize = file.tellp();
    if(size() == fileSize) {
      cout << "Successfully wrote file of size " << fileSize  << endl;
      return true;
    } else {
      cout << "Unexpected file size." << endl;
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
  
  cout << "Creating char file_array of size " << (1.0*sizeof(char)*in_block_size)/pow(1024,2) << " MB." << endl;
  char *file_data = (char *)malloc(in_block_size);  

  // read one block of input data  
  // get size of file
  if(!input_rands) {
    cout << "attempting to read file " << input_filename.c_str() << endl; 
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

    random_device rd;
    mt19937_64 gen(rd());
    gen.seed(1234);
    uniform_int_distribution<uint64_t> dis;
    for (int i = 0; i < n_rand; i++) input_rand[i] = dis(gen);
    //for (int i = 0; i < n_rand; i++) input_rand[i] = (uint64_t)1234;
    memcpy(file_data, (void*)input_rand, n_rand);
    free(input_rand);
  }
  
  // Start dsaX program
  //---------------------------------------
  timer::Timer<chrono::microseconds, chrono::high_resolution_clock> test_timer;

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
  cout << "Creating char input array of size " << input_size << " bytes." << endl;
  void *input_data = dsaXHostRegister(input_size);
  // Populate with random data. Each stream has the same data
  // To ensure the concurrency does not pollute accross streams. 
  for (int i = 0; i<n_streams; i++) memcpy((char*)input_data + i * in_block_size, file_data, in_block_size);

  // Create GPU registered output array
  uint64_t output_size = n_streams * sizeof(float) * NBASE*NCHAN_PER_PACKET*2*2;
  cout << "Creating float output_array of size " << output_size << " bytes." << endl;
  void *output_data = dsaXHostRegister(output_size);

  /*
  float *A = (float*)dsaXHostRegister(2*sizeof(float)*96*512);
  float *B = (float*)dsaXHostRegister(2*sizeof(float)*96*512);
  float *C = (float*)dsaXHostRegister(2*sizeof(float)*96*96);
  promoteComplexCharToFloat(A, file_data, 512, 96);
  promoteComplexCharToFloat(B, file_data, 512, 96);  
  host_MdagM_gemm(A, B, C, 96, 96, 512); 
  */
    
  // Ensure test output array is zero
  memset(output_data, 0, output_size);
  
  cout << "Total input size = " << (1.0 * input_size)/pow(1024,3) << " GB." << endl;
  cout << "Expected output size = " << (1.0 * output_size)/pow(1024,3) << " GB." << endl;
  
  test_timer.start();  
  correlator->compute(output_data, input_data);
  test_timer.stop();

  float frob_norm = test_hermiticity((float*)output_data, 96, 96);
  cout << "Frobenius norm = " << frob_norm << endl;

  
  //cout << "Output peek " << endl;
  float *p = (float*)output_data;
  for(int i=0; i<8; i++) cout << "output[" << i << "] = " << p[i] << endl;
  
  if(write_output) {
    fout = fopen(output_filename.c_str(),"ab");
    fwrite((unsigned char *)output_data, sizeof(unsigned char *), sizeof(float)*output_size, fout);
    fclose(fout);
  }
  
  delete correlator;
  dsaXEnd();

  cout << "Test time = " << (1.0*test_timer.elapsed().count())/(1e6) << " seconds. " << endl;
  
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
    cout << "Failed to read the file." << endl;
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
  for (int i=0; i<NBASE*NCHAN_PER_PACKET*2*2; i++) if(output_data[i] != 0) cout << "output " << i << " = " << output_data[i] << endl;
  //for (int i=0; i<8; i++) cout << "output " << i << " = " << output_data[i] << endl; 

  if (!binaryFileVector.saveToFile("output.dat")) {
    cout << "Failed to write a file." << endl;
    return 0;
  } else {
    cout << "Successfully wrote file." << endl;
  }
  
  
  fout = fopen("output.dat","wb");
  fwrite((float *)output_data, sizeof(float), NBASE*NCHAN_PER_PACKET*2*2, fout);
  fclose(fout);
  */
      

}
