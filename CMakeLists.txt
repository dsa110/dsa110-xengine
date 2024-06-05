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
project("DSA_XENGINE" VERSION 1.0.0 LANGUAGES CXX CUDA C)

# DSA_XENGINE may be built to run using CUDA. Future version may be
# written for HIP or SYCL, which we call the
# Target type. By default, the target is CUDA.
if(DEFINED ENV{DSA_XENGINE_TARGET})
  set(DEFTARGET $ENV{DSA_XENGINE_TARGET})
else()
  set(DEFTARGET "CUDA")
endif()

set(VALID_TARGET_TYPES CUDA) #HIP SYCL
set(DSA_XENGINE_TARGET_TYPE
  "${DEFTARGET}"
  CACHE STRING "Choose the type of target, options are: ${VALID_TARGET_TYPES}")
set_property(CACHE DSA_XENGINE_TARGET_TYPE PROPERTY STRINGS CUDA)

# CUDA specific part of CMakeLists
#set(CMAKE_CUDA_EXTENSIONS OFF)
find_package(CUDAToolkit REQUIRED)

if(DEFINED ENV{DSA_XENGINE_GPU_ARCH})
  set(DSA_XENGINE_DEFAULT_GPU_ARCH $ENV{DSA_XENGINE_GPU_ARCH})
else()
  set(DSA_XENGINE_DEFAULT_GPU_ARCH sm_70)
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

string(TOUPPER ${DSA_XENGINE_TARGET_TYPE} CHECK_TARGET_TYPE)
list(FIND VALID_TARGET_TYPES ${CHECK_TARGET_TYPE} TARGET_TYPE_VALID)

if(TARGET_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid DSA_XENGINE_TARGET_TYPE type! Valid target types are:" "${VALID_TARGET_TYPES}")
endif()

# Git
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

# EXTERNALS
include(FetchContent)
# Get psrdada dependency
option(DSA_XENGINE_DOWNLOAD_PSRDADA "Download and build PSRDada" ON)
if(DSA_XENGINE_DOWNLOAD_PSRDADA) 
  FetchContent_Declare(
    PSRDada
    GIT_REPOSITORY git://git.code.sf.net/p/psrdada/code
    #GIT_TAG        008afa70393ae2df11efba0cc8d0b95cda599c02
    )
  FetchContent_MakeAvailable(PSRDada)
endif()

# Get XGPU dependency
option(DSA_XENGINE_DOWNLOAD_XGPU "Download and build XGPU" ON)
if(DSA_XENGINE_DOWNLOAD_XGPU) 
  FetchContent_Declare(
    xGPU
    GIT_REPOSITORY https://github.com/cpviolator/xGPU.git
    #GIT_TAG        13b7fff1eac497236eb9c38e179aed3b532a88f2
    )
  FetchContent_MakeAvailable(XGPU)
endif()

# Get TCC dependency
option(DSA_XENGINE_DOWNLOAD_TCC "Download and build TCC" ON)
if(DSA_XENGINE_DOWNLOAD_TCC) 
  FetchContent_Declare(
    TCC
    GIT_REPOSITORY https://git.astron.nl/RD/tensor-core-correlator
    #GIT_TAG        11d8a4a504d7073a2a33b81e1e387b12e58a420c
    )
  FetchContent_MakeAvailable(TCC)
endif()
add_custom_command(
  OUTPUT "file.txt"
  WORKING_DIRECTORY ${TCC_SOURCE_DIR}
  COMMAND "sed -i 's/libtcc\///g' libtcc/*.h libtcc/*.cc"
  )  


# Add src
add_subdirectory(src)

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
