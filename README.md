# dsa110-xengine

This repo contains code used for the DSA X-engine. The requirements are to:
 - capture SNAP F-engine packets on an ethernet interface, and place them in a psrdada buffer
 - run the xgpu kernel to correlate all inputs present
 - accumulate the visibilities to produce a data stream for calibration
 - beamform from the visibilities to produce search-data streams

### v0.1

This is a first go version at capture, xgpu, and accumulation. Logging is simply to stdout and stderr, and M&C is accomplished with a single socket connection and logging.


