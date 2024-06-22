enable_language(CUDA)

include_directories(../include)

if(DSA_XENGINE_ENABLE_PSRDADA)
  include_directories(${PSRDada_SOURCE_DIR}/src)
  set(PSRDada_LIB ${PSRDada_BINARY_DIR}/src/libpsrdada.so)
endif()

if(DSA_XENGINE_ENABLE_XGPU) 
  include_directories(${xGPU_SOURCE_DIR}/src)
  set(XGPU_LIB ${xGPU_BINARY_DIR}/src/libxgpu.a)
endif()

if(DSA_XENGINE_ENABLE_CUTLASS) 
  include_directories(${NvidiaCutlass_DIR}/../../../include)
  include_directories(${NvidiaCutlass_DIR}/../../../include/cutlass/util)
  set(NvidiaCutlass_LIB ${NvidiaCutlass_DIR}/../../../lib64/libcutlass.so)

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

# DSA Fast Time Domain
#---------------------
add_library(dsa OBJECT dsaX_beamformer_correlator.cu)

add_executable(dsaX_beamformer_correlator dsaX_beamformer_correlator.cu)
target_link_libraries(dsaX_beamformer_correlator ${CUDA_cublas_LIBRARY} ${PSRDada_LIB})
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
  dsaX_beamformer_correlator
  RUNTIME DESTINATION
  bin
  )
#-----------------------------
