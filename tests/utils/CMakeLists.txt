# install step for utils
#------------------------------
set(DSA_XENGINE_UTILS
  # cmake-format: sortable
  gen_packet.py
  get_rms_packet.py
  get_rms.py
  sockets.py
  )
install(FILES ${DSA_XENGINE_UTILS} DESTINATION utils)
#------------------------------
