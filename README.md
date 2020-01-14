# dsa110-xengine

This repo contains code used for the DSA X-engine. The requirements are to:
 - capture SNAP F-engine packets on an ethernet interface, and place them in a psrdada buffer
 - run the xgpu kernel to correlate all inputs present
 - accumulate the visibilities to produce a data stream for calibration
 - beamform from the visibilities to produce search-data streams

### v0.1

This is a first go version at capture, xgpu, and accumulation. Logging is simply to stdout and stderr, and M&C is accomplished with a single socket connection and logging.

### v1.0 description

* 32 antennas, 2 pol, 6144 channels. 
* Per server, capture up to 30 antennas (dual pol) in 1536 channels = 22.5 Gbps. Run single instance of xgpu natively integrating by x32, but integrate further by 128. 

Output ordering from dsaX_xgpu is (baseline, frequency, pol, r/i)


