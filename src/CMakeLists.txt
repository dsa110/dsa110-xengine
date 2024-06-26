include_directories(${CMAKE_SOURCE_DIR}/include)

if(DSA_XENGINE_TARGET_TYPE STREQUAL CUDA)
  add_compile_definitions(DSA_XENGINE_TARGET_CUDA)
endif()

if(DSA_XENGINE_TARGET_TYPE STREQUAL CPU)
  add_compile_definitions(DSA_XENGINE_TARGET_CPU)
endif()

# DSA Fast Time Domain library
#-----------------------------
set(DSAX_OBJS
  dsaX_cuda_interface.cu
  dsaX_cublas_interface.cu
  dsaX_magma_interface.cu
  dsaX_blas_interface.cpp
  dsaX_beamformer.cpp
  dsaX_correlator.cpp
  dsaX_interface.cpp
  dsaX_utils.cpp
  dsaX_psrdada_utils.cpp
  )

# split source into cu and cpp files
foreach(item ${DSAX_OBJS})
  string(REGEX MATCH ".+\\.cu$" item_match ${item})
  if(item_match)
    list(APPEND DSAX_CU_OBJS ${item})
  endif(item_match)
endforeach(item ${DSAX_OBJS})

list(REMOVE_ITEM DSAX_OBJS ${DSAX_CU_OBJS})

# DSAX_CU_OBJS should contain all cuda files now and DSAX_OBJS all cpp.
# If we have a git version, make version.cpp depend on git head so that it is
# rebuilt if the git sha changed
if(GITVERSION)
  find_path(
    DSAX_GITDIR NAME HEAD
    PATHS ${CMAKE_SOURCE_DIR}/.git/logs
    NO_DEFAULT_PATH)
  include(AddFileDependencies)
  if(DSAX_GITDIR)
    add_file_dependencies(version.cpp ${DSAX_GITDIR}/HEAD)
  endif()
endif()
mark_as_advanced(DSAX_GITDIR)

# generate a cmake object library for all cpp files first                                                                                                                                                           
add_library(dsax_cpp OBJECT ${DSAX_OBJS})

if(DSA_XENGINE_BUILD_SHAREDLIB)
  set_target_properties(dsax_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_library(dsax SHARED)
else()
  add_library(dsax STATIC)
endif()
add_library(DSA_XENGINE::dsax ALIAS dsax)

# make one library                                                                                                                                                                                                  
target_sources(dsax PRIVATE $<TARGET_OBJECTS:dsax_cpp> ${DSAX_CU_OBJS})

if(CUDAToolkit_FOUND)
  target_link_libraries(dsax INTERFACE CUDA::cudart_static ${CUDA_cublas_LIBRARY})
endif()

if(DSA_XENGINE_ENABLE_PSRDADA)
  include_directories(${PSRDada_SOURCE_DIR}/src)
  set(PSRDada_LIB ${PSRDada_BINARY_DIR}/src/libpsrdada.so)
  target_link_libraries(dsax PUBLIC ${PSRDada_LIB})
endif()

if(DSA_XENGINE_ENABLE_XGPU) 
  include_directories(${xGPU_SOURCE_DIR}/src)
  set(XGPU_LIB ${xGPU_BINARY_DIR}/src/libxgpu.a)
  target_link_libraries(dsax PUBLIC ${XGPU_LIB})
endif()

if(DSA_XENGINE_ENABLE_CUTLASS) 
  include_directories(${NvidiaCutlass_DIR}/../../../include)
  include_directories(${NvidiaCutlass_DIR}/../../../include/cutlass/util)
  set(NvidiaCutlass_LIB ${NvidiaCutlass_DIR}/../../../lib64/libcutlass.so)
  target_link_libraries(dsax PUBLIC ${NvidiaCutlass_LIB})
  
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
#---------------------

# install step for libraray
#-----------------------------
install(TARGETS
  # cmake-format: sortable
  dsax
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

if(CUDAToolkit_FOUND)
  #add_executable(dsaX_beamformer_correlator_exe dsaX_beamformer_correlator_exe.cu)
  #target_link_libraries(dsaX_beamformer_correlator_exe PUBLIC dsax ${CUDA_cublas_LIBRARY} ${PSRDada_LIB})
endif()
