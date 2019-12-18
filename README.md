# dsa110-xengine

This repo contains code used for the DSA X-engine. The requirements are to:
 - capture SNAP F-engine packets on an ethernet interface, and place them in a psrdada buffer
 - run the xgpu kernel to correlate all inputs present
 - accumulate the visibilities to produce a data stream for calibration
 - beamform from the visibilities to produce search-data streams
 
Each branch pertains to a different SNAP firmware version. A whole-number increment in the version number implies a tested release with capabilities as advertised. 
