#DMH: fix include path
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CLI11_SOURCE_DIR}/include/CLI)

add_executable(dsaX_correlator_test dsaX_correlator_test.cpp)
target_link_libraries(dsaX_correlator_test dsax)
