enable_language(CUDA)

include_directories(..//include)
include_directories(${PSRDada_SOURCE_DIR}/src)
include_directories(${xGPU_SOURCE_DIR}/src)
include_directories(${NvidiaCutlass_DIR}/../../../include)
include_directories(${NvidiaCutlass_DIR}/../../../include/cutlass/util)

set(PSRDada_LIB ${PSRDada_BINARY_DIR}/src/libpsrdada.so)
set(XGPU_LIB ${xGPU_BINARY_DIR}/src/libxgpu.a)
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

# DSA Fast Time Domain
#---------------------
add_executable(dsaX_bfCorr dsaX_bfCorr.cu)
target_link_libraries(dsaX_bfCorr ${CUDA_cublas_LIBRARY} ${PSRDada_LIB})
#---------------------

# install step for header files
#------------------------------
set(DSA_XENGINE_HEADERS
  # cmake-format: sortable
  dsaX_capture.h
  dsaX_capture_manythread.h
  dsaX_capture_pcap.h
  dsaX_def.h
  dsaX_cutlass_interface.h
  )
install(FILES ${DSA_XENGINE_HEADERS} DESTINATION include)
#------------------------------

# install step for executables
#-----------------------------
install(TARGETS
  # cmake-format: sortable
  dsaX_bfCorr
  RUNTIME DESTINATION
  bin
  )
#-----------------------------
