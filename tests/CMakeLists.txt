#DMH: fix include path
include_directories(../include)
include_directories(${CLI11_SOURCE_DIR}/include/CLI)
add_executable(dsaX_correlator_test dsaX_correlator_test.cpp)

