####################################################################################
# START 1. Basic setup for cmake
####################################################################################
# basic setup for cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Disable gnu exentions
set(CMAKE_CXX_EXTENSIONS ON)

# Define the project
project("DSA_XENGINE" VERSION 1.0.0 LANGUAGES C CXX)

# For GCC 8 and lower, set -pthread flag manually
set(CMAKE_C_FLAGS "-pthread")
set(CMAKE_CXX_FLAGS "-pthread")

# add a directory for cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# DSA_XENGINE may be built to run using CUDA or CPU. Future version may be
# written for HIP or SYCL, which we call the
# Target type. By default, the target is CUDA.
#---------------------------------------------

# Set by environment variable if visible
if(DEFINED ENV{DSA_XENGINE_TARGET})
  set(DEFTARGET $ENV{DSA_XENGINE_TARGET})
else()
  set(DEFTARGET "CUDA")
endif()

set(VALID_TARGET_TYPES CUDA CPU) #HIP SYCL
set(DSA_XENGINE_TARGET_TYPE "${DEFTARGET}" CACHE STRING "Choose the type of target, options are: ${VALID_TARGET_TYPES}")
set_property(CACHE DSA_XENGINE_TARGET_TYPE PROPERTY STRINGS "CUDA" "CPU") # HIP SYCL

string(TOUPPER ${DSA_XENGINE_TARGET_TYPE} CHECK_TARGET_TYPE)
list(FIND VALID_TARGET_TYPES ${CHECK_TARGET_TYPE} TARGET_TYPE_VALID)

if(TARGET_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid DSA_XENGINE_TARGET_TYPE type! Valid target types are:" "${VALID_TARGET_TYPES}")
endif()

# Git helpers
#------------
find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} show
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE IS_GIT_REPOSIITORY
    OUTPUT_QUIET ERROR_QUIET)
  if(${IS_GIT_REPOSIITORY} EQUAL 0)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --abbrev=0
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITTAG
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    # we use git rev-list and pipe that through wc here. Newer git versions support --count as option to rev-list but
    # that might not always be available
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list ${GITTAG}..HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMAND wc -l
      OUTPUT_VARIABLE GITCOUNT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --match 1 --always  --long --dirty
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITVERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif(GIT_FOUND)


option(DSA_XENGINE_BUILD_ALL_TESTS "build tests by default" ON)
option(DSA_XENGINE_INSTALL_ALL_TESTS "install tests by default" ON)
option(DSA_XENGINE_BUILD_SHAREDLIB "build dsaXengine as a shared lib" ON)


# Use ExternalProject_Add for libtcc (borks with FetchContent)
# Use ExternalProject_Add for CUTLASS (long build time, version 2.11.0 for sm_8x arch)
include(ExternalProject)

# Use FetchContent for lightweight dependencies
include(FetchContent)

# CUDA based dependencies and options
#------------------------------------
if(DSA_XENGINE_TARGET_TYPE STREQUAL CUDA)

  # CUDA specific part of CMakeLists
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # Get GPU architecture from environmen, or set default (sm_80)
  if(DEFINED ENV{DSA_XENGINE_GPU_ARCH})
    set(DSA_XENGINE_DEFAULT_GPU_ARCH $ENV{DSA_XENGINE_GPU_ARCH})
  else()
    set(DSA_XENGINE_DEFAULT_GPU_ARCH sm_80)
  endif()
  
  if(NOT DSA_XENGINE_GPU_ARCH)
    message(STATUS "Building DSA_XENGINE for GPU ARCH " "${DSA_XENGINE_DEFAULT_GPU_ARCH}")
  endif()
  
  set(DSA_XENGINE_GPU_ARCH
    ${DSA_XENGINE_DEFAULT_GPU_ARCH}
    CACHE STRING "set the GPU architecture (sm_60, sm_70, sm_80 sm_90)")
  set_property(CACHE DSA_XENGINE_GPU_ARCH PROPERTY STRINGS sm_60 sm_70 sm_80 sm_90)
  set(DSA_XENGINE_GPU_ARCH_SUFFIX
    ""
    CACHE STRING "set the GPU architecture suffix (virtual, real). Leave empty for no suffix.")
  set_property(CACHE DSA_XENGINE_GPU_ARCH_SUFFIX PROPERTY STRINGS "real" "virtual" " ")
  #set(CMAKE_CUDA_ARCHITECTURES ${DSA_XENGINE_GPU_ARCH})
  #mark_as_advanced(DSA_XENGINE_GPU_ARCH_SUFFIX)
  #mark_as_advanced(CMAKE_CUDA_ARCHITECTURES)
  
  # Set CUDA based methods and dependencies
  #----------------------------------------
  
  # This is the default GPU method
  option(DSA_XENGINE_ENABLE_CUBLAS "Use cuBLAS for correlatorss" ON)

  # All other GPU methods can be enabled at compile time and
  # toggled for use at run time, if enabled.
  
  # Get TCC dependency
  option(DSA_XENGINE_ENABLE_TCC "Use TensorCoreCorrelators for correlatorss" OFF)
  if(DSA_XENGINE_ENABLE_TCC)
    add_compile_definitions(DSA_XENGINE_ENABLE_TCC)
    option(DSA_XENGINE_DOWNLOAD_TCC "Download, build, link (and install) TCC" OFF)
    if(DSA_XENGINE_DOWNLOAD_TCC)
      ExternalProject_Add(TCC
	GIT_REPOSITORY https://git.astron.nl/RD/tensor-core-correlator
	#GIT_TAG        11d8a4a504d7073a2a33b81e1e387b12e58a420c
	CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
	)
    else()
      find_package(libtcc REQUIRED)
    endif()
  endif()
  
  # Get CUTLASS dependency
  option(DSA_XENGINE_ENABLE_CUTLASS "Use CUTLASS for GEMMs" OFF)
  if(DSA_XENGINE_ENABLE_CUTLASS)
    add_compile_definitions(DSA_XENGINE_ENABLE_CUTLASS)
    option(DSA_XENGINE_DOWNLOAD_CUTLASS "Download, build (only the required kernels) link (and install) CUTLASS" OFF)
    if(DSA_XENGINE_DOWNLOAD_CUTLASS)
      # Custom CUTLASS build
      ExternalProject_Add(NvidiaCutlass
	GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
	GIT_TAG 7d49e6c7e2f8896c47f586706e67e1fb215529dc
	CMAKE_ARGS
	"-DCUTLASS_NVCC_ARCHS_ENABLED=89"
	"-DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_*gemm_planar_complex"
	"-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
	)
    else()
      find_package(NvidiaCutlass REQUIRED)
    endif()
  endif()
  
  # Get MAGMA dependency
  option(DSA_XENGINE_ENABLE_MAGMA "Use MAGMA for GEMMs" OFF)
  if(DSA_XENGINE_ENABLE_MAGMA)
    add_compile_definitions(DSA_XENGINE_ENABLE_MAGMA)
    option(DSA_XENGINE_DOWNLOAD_MAGMA "Download, build (only the required kernels) link (and install) MAGMA" OFF)
    if(DSA_XENGINE_DOWNLOAD_MAGMA)
      # Custom MAGMA build
      ExternalProject_Add(Magma
	URL https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.8.0.tar.gz
	CMAKE_ARGS
	"-DMAGMA_ENABLE_CUDA=ON"
	"-DGPU_TARGET=sm_80"
	"-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
	)
    else()
      find_package(Magma REQUIRED)
    endif()
  endif()
  
  # Get XGPU dependency (fix install)
  option(DSA_XENGINE_ENABLE_XGPU "Use xGPU for correlatorss" OFF)
  if(DSA_XENGINE_ENABLE_XGPU)
    add_compile_definitions(DSA_XENGINE_ENABLE_XGPU)
    option(DSA_XENGINE_DOWNLOAD_XGPU "Download and build xGPU" OFF)
    if(DSA_XENGINE_DOWNLOAD_XGPU)
      # Download, build and install
      FetchContent_Declare(    
	xGPU
	GIT_REPOSITORY https://github.com/cpviolator/xGPU.git
	#GIT_TAG        13b7fff1eac497236eb9c38e179aed3b532a88f2
	)
      FetchContent_MakeAvailable(XGPU)
    else()
      # Find and link to local install
      find_package(xGPU REQUIRED)
    endif()
  endif()

endif() # CUDA functionality

# Get CPU based dependencies
# Get OPENBLAS dependency
option(DSA_XENGINE_ENABLE_OPENBLAS "Use OPENBLAS for GEMMs" OFF)
if(DSA_XENGINE_ENABLE_OPENBLAS)
  add_compile_definitions(DSA_XENGINE_ENABLE_OPENBLAS)
  option(DSA_XENGINE_DOWNLOAD_OPENBLAS "Download, build, link, and install OPENBLAS" OFF)
  if(DSA_XENGINE_DOWNLOAD_OPENBLAS)
    # Custom OPENBLAS build
    ExternalProject_Add(Openblas
      GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
      GIT_TAG ce3f668
      CMAKE_ARGS
      #"-DOPENBLAS_ENABLE_CUDA=ON"
      #"-DGPU_TARGET=sm_80"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
      )
  else()
    find_package(Openblas REQUIRED)
  endif()
endif()

# Get psrdada dependency
option(DSA_XENGINE_ENABLE_PSRDADA "Use PSRDada for IO" ON)
option(DSA_XENGINE_DOWNLOAD_PSRDADA "Download and build PSRDada" ON)
if(DSA_XENGINE_DOWNLOAD_PSRDADA)
  # Download, build and install
  FetchContent_Declare(
    PSRDada
    GIT_REPOSITORY git://git.code.sf.net/p/psrdada/code
    )
  FetchContent_MakeAvailable(PSRDada)
else()
  # Find and link to local install
  find_package(PSRDada REQUIRED)
endif()

# Get HDF5 dependency
option(DSA_XENGINE_ENABLE_HDF5 "Use HDF5 for data IO" OFF)
if(DSA_XENGINE_ENABLE_HDF5)
  option(DSA_XENGINE_DOWNLOAD_HDF5 "Download and build HDf5" OFF)
  if(DSA_XENGINE_DOWNLOAD_HDF5)
    # Download, build and install
    FetchContent_Declare(
      HDF5
      GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
      GIT_TAG 5794814
      )
    FetchContent_MakeAvailable(HDF5)
  else()
    # Find and link to local install
    find_package(HDF5 REQUIRED)
  endif()
endif()

# Get CLI11 dependency
# FIX ME: get static .hpp version and ship with package
option(DSA_XENGINE_ENABLE_CLI11 "Enable CLI11 (required)" ON)
if(DSA_XENGINE_ENABLE_CLI11)
  option(DSA_XENGINE_DOWNLOAD_CLI11 "Download and build CLI11" ON)
  if(DSA_XENGINE_DOWNLOAD_CLI11)
    # Download, build and install
    FetchContent_Declare(
      CLI11
      GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
      GIT_TAG main
      )
    FetchContent_MakeAvailable(CLI11)
  else()
    # Find and link to local install
    find_package(CLI11 REQUIRED)
  endif()
endif()


# Get ZFP dependency
option(DSA_XENGINE_ENABLE_ZFP "Enable ZFP" OFF)
if(DSA_XENGINE_ENABLE_ZFP)
  option(DSA_XENGINE_DOWNLOAD_ZFP "Download and build ZFP" OFF)
  if(DSA_XENGINE_DOWNLOAD_ZFP)
    # Download, build and install
    FetchContent_Declare(
      ZFP
      GIT_REPOSITORY https://github.com/LLNL/zfp.git
      GIT_TAG f40868a
      )
    FetchContent_MakeAvailable(ZFP)
  else()
    # Find and link to local install
    find_package(ZFP REQUIRED)
  endif()
endif()

# Get Timer dependency https://github.com/cpp-core/timer.git
# Get timer dependency
option(DSA_XENGINE_ENABLE_TIMER "Enable timer" ON)
if(DSA_XENGINE_ENABLE_TIMER)
  option(DSA_XENGINE_DOWNLOAD_TIMER "Download and build timer" ON)
  if(DSA_XENGINE_DOWNLOAD_TIMER)
    # Download, build and install
    FetchContent_Declare(
      TIMER
      GIT_REPOSITORY https://github.com/cpp-core/timer.git
      GIT_TAG main
      )
    FetchContent_MakeAvailable(TIMER)
  else()
    # Find and link to local install
    find_package(TIMER REQUIRED)
  endif()
endif()

# Add src, include, tests, and legacy
add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(tests)
option(DSA_XENGINE_BUILD_LEGACY "Build legacy code (will not install if built)" OFF)
if(DSA_XENGINE_BUILD_LEGACY)
  add_subdirectory(legacy)
endif()

# Install project cmake targets
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_NAME}-config-version.cmake
  VERSION ${DSA_XENGINE_VERSION}
  COMPATIBILITY AnyNewerVersion
  )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
