#include <iostream>
#include <utils.h>
#include <cuda_handles.h>

using namespace std;

#ifdef DSA_XENGINE_TARGET_CUDA

// CUDA stream handler functions
//-------------------------
void init_streams(unsigned int n_streams) {

  if(n_streams < 2 || n_streams > 9) {
    cout << "dsaX Error: Must have at least 2 and fewer than 9 streams, requested " << n_streams << endl;
    exit(0);
  }
  
  if(!stream_init) {
    streams.reserve(n_streams);
    for (auto &s : streams) cudaStreamCreate(&s);
    /*
      int greatestPriority;
      int leastPriority;
    
      // Query the device to get its built in priority range
      // For CUDA, lower numerical values indicate higher priority
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
      for (int i=0; i<Nstream-1; i++) {
      
      // Set streams 0 to Nstream-1 to have highest priority
      cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, greatestPriority);
      }
    
      // Set stream Nstream - 1 to have lowest priority
      cudaStreamCreateWithPriority(&streams[Nstream - 1], cudaStreamDefault, leastPriority);
    */    
    stream_init = true;
  }
}

void destroy_streams() {
  if (stream_init) {
    for (auto &s : streams) cudaStreamDestroy(s);
    stream_init = false;
  } else {
    cout << "dsaX Warning: streams not initialized. Please call dsaXInitStreams(n) before destroying streams." << endl;
  }
}

cudaStream_t get_stream(unsigned int i) {  
  if(!stream_init) {
    cout << "dsaX Error: streams not initialized. Please call dsaXInitStreams(n) before getting stream." << endl;
    exit(0);
  }
  return streams[i];
}

#else

// Empty error out functions if called from non
// CUDA terget enabled builds
void init_streams() cout << "dsaX Error: CUDA target not build" << endl; exit(0);
void destroy_streams() cout << "dsaX Error: CUDA target not build" << endl; exit(0);
#endif
