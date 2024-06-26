enable_language(CUDA)

# install step for header files
#------------------------------
set(DSA_XENGINE_HEADERS
  # cmake-format: sortable
  dsaX.h
  dsaX_def.h
  dsaX_ftd.h  
  dsaX_cuda_interface.h
  dsaX_cuda_headers.h
  dsaX_capture.h
  dsaX_capture_manythread.h
  dsaX_capture_pcap.h
  dsaX_cutlass_interface.h
  )
install(FILES ${DSA_XENGINE_HEADERS} DESTINATION include)
#------------------------------
