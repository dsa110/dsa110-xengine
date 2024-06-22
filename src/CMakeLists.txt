enable_language(CUDA)

include_directories(../include)

# DSA Fast Time Domain library
#-----------------------------
add_library(dsa SHARED dsaX_cuda_interface.cu dsaX_blas_interface.cu dsaX_beamformer_correlator.cu)

if(CUDAToolkit_FOUND)
  target_link_libraries(dsa CUDA::cudart)
endif()

if(DSA_XENGINE_ENABLE_PSRDADA)
  include_directories(${PSRDada_SOURCE_DIR}/src)
  set(PSRDada_LIB ${PSRDada_BINARY_DIR}/src/libpsrdada.so)
  target_link_libraries(dsa ${PSRDada_LIB})
endif()

if(DSA_XENGINE_ENABLE_XGPU) 
  include_directories(${xGPU_SOURCE_DIR}/src)
  set(XGPU_LIB ${xGPU_BINARY_DIR}/src/libxgpu.a)
  target_link_libraries(dsa PUBLIC ${XGPU_LIB})
endif()

if(DSA_XENGINE_ENABLE_CUTLASS) 
  include_directories(${NvidiaCutlass_DIR}/../../../include)
  include_directories(${NvidiaCutlass_DIR}/../../../include/cutlass/util)
  set(NvidiaCutlass_LIB ${NvidiaCutlass_DIR}/../../../lib64/libcutlass.so)
  target_link_libraries(dsa PUBLIC ${NvidiaCutlass_LIB})
  
  # Some simple CUTLASS examples to test linking/benching
  #------------------------------------------------------
  add_executable(planar_complex planar_complex.cu)
  target_link_libraries(planar_complex ${NvidiaCutlass_LIB})
  
  add_executable(10_planar_complex 10_planar_complex.cu)
  target_link_libraries(10_planar_complex ${NvidiaCutlass_LIB})
  
  add_executable(11_planar_complex_array 11_planar_complex_array.cu)
  target_link_libraries(11_planar_complex_array ${NvidiaCutlass_LIB})
  #------------------------------------------------------

  # DSA Fast Time Domain CUTLASS interface
  #---------------------------------------
  add_executable(dsaX_cutlass_interface dsaX_cutlass_interface.cu)
  target_link_libraries(dsaX_cutlass_interface ${NvidiaCutlass_LIB})
  #---------------------------------------  
endif()

if(CUDAToolkit_FOUND)
  #add_executable(dsaX_beamformer_correlator dsaX_beamformer_correlator.cu)
  #target_link_libraries(dsaX_beamformer_correlator ${dsa} ${CUDA_cublas_LIBRARY} ${PSRDada_LIB})
endif()
#---------------------

# install step for libraray
#-----------------------------
install(TARGETS
  # cmake-format: sortable
  dsa
  LIBRARY DESTINATION
  lib
  )
#-----------------------------


# install step for executables
#-----------------------------
install(TARGETS
  # cmake-format: sortable
  #dsaX_beamformer_correlator
  RUNTIME DESTINATION
  bin
  )
#-----------------------------
