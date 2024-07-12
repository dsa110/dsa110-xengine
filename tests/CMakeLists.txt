#DMH: fix include path
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CLI11_SOURCE_DIR}/include/CLI)

add_library(dsaX_tests command_line_params.cpp)

add_executable(correlator_test correlator_test.cpp)
target_link_libraries(correlator_test dsaX dsaX_tests)
