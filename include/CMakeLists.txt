enable_language(CUDA)

# install step for header files
#------------------------------
set(DSA_XENGINE_HEADERS
  # cmake-format: sortable
  dsaX.h
  dsaX_def.h
  fast_time_domain.h  
  cuda_interface.h
  cuda_handles.h
  cuda_headers.h
  dsaX_capture.h
  dsaX_capture_manythread.h
  dsaX_capture_pcap.h
  cutlass_interface.h
  )
install(FILES ${DSA_XENGINE_HEADERS} DESTINATION include)
#------------------------------
