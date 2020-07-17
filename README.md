# dsa110-xengine

This repo contains code used for the DSA X-engine. The requirements are to:
 - capture SNAP F-engine packets on an ethernet interface, and place them in a psrdada buffer
 - run the xgpu kernel to correlate all inputs present
 - accumulate the visibilities to produce a data stream for calibration
 - beamform from the visibilities to produce search-data streams
 - corner turn the beams
 
 ## v0.9
 
 This version contains all sorts of good stuff. Mainly, semi-tested routines to fulfill the following functionality for 64 dual-pol antennas:
  - produce visibilities with 0.134217728s integrations and 6144 channels of width 250/8192 MHz. Dual pol.
  - form 256 Stokes-I beams from the voltage streams with 1.048576ms integrations and 768 channels of width 250/1024 MHz 
  - corner-turn the beams to deliver (to mbheimdall) concatenated blocks of size [64 beams, 4096 integrations, 1024 channels]
  
 To compile, simply clone and run "make" in the src dir. Edit the makefile to sort out dependencies.
 
 ### notes on architecture
 
 All dada buffer names, and most defining constants, can be found in `dsaX_def.h`.
 
 `dsaX_capture` captures udp packets from multiple SNAPs and places them in a single dada buffer (`CAPTURE_BLOCK_KEY`). Relies on important constants like `NCHANG`, `NSNAPS`, `CHOFF`. The UDP packets have format 64-bit header, then [3 antennas, 384 channels, 2 times, 2 pols, 4-bit complex]. 
 
 `dsaX_fake` can be used in place of `dsaX_capture`, when primed with a junkdb driving the `TEST_BLOCK_KEY` buffer. Need to edit the `out_key` in the code to choose the output buffer.  
 
 `dsaX_split` comes next, reading from `CAPTURE_BLOCK_KEY` into `CAPTURED_BLOCK_KEY` and `REORDER_BLOCK_KEY2`. The latter is filled with permuted data to feed the beamformer. This also prints some useful stats on the input datastream to syslog, specifically the per-input rms values. 
 
 ---
 
 On the cross-correlation side, `dsaX_reorder_raw` reads from `CAPTURED_BLOCK_KEY`, fluffing the data to 8-bit complex and reordering for input to xgpu. many threads and avx-512 instructions are used for this. Writes to `REORDER_BLOCK_KEY`.
 
`TEST_BLOCK_KEY`, `CAPTURE_BLOCK_KEY`, `CAPTURED_BLOCK_KEY` should have sizes 2048 packets x NANT ants x 384 chans x 2 times x 2 pols = 198,180,864 bytes.  `REORDER_BLOCK_KEY`, `REORDER_BLOCK_KEY2` should have sizes 2048 packets x 64 ants x 384 chans x 2 times x 2 pols x R/I = 402,653,184 bytes. 

The `dsaX_xgpu` code does the cross-correlation on data in `REORDER_BLOCK_KEY`, and pipes the data straight out to `XGPU_BLOCK_KEY`. The command `xgpuinfo` should result in the following output.

Number of polarizations: 2
Number of stations: 64
Number of baselines: 2080
Number of frequencies: 768
Number of time samples per GPU integration: 2048
Number of time samples per transfer to GPU: 128
Type of ComplexInput components: 8 bit integers
Type of computation: FP32 multiply, FP32 accumulate
Number of ComplexInput elements in GPU input vector: 201326592
Number of ComplexInput elements per transfer to GPU: 12582912
Number of Complex elements in GPU output vector: 6389760
Number of Complex elements in reordered output vector: 6389760
Output matrix order: triangular
Shared atomic transfer size: 4
Complex block size: 1

This implies compilation as follows:
make CUDA_ARCH=sm_75 NPOL=2 NSTATION=64 NFREQUENCY=768 NTIME_PIPE=128 NTIME=2048
Be sure to check the output matrix order

The cross-correlation side finishes with `dsaX_writevis`, which is test code to write visibilities to disk. `XGPU_BLOCK_KEY` should contain a single reordered output vector (6389760 complex FP32 elements), corresponding to 51,118,080 bytes. 

---

On the beamformer side, the output of `dsaX_split` is piped into `dsaX_beamformer`. This is fantastic code that uses the tensor cores (and a few other tricks) in the Turing 104 GPUs to do the beamforming on data in `REORDER_BLOCK_KEY2`, piping the data into `BF_BLOCK_KEY`. The latter needs to have size 128 times x 256 beams x 48 chans = 1,572,864. 

Finally, `dsaX_dbnic` and `dsaX_nicdb` implement the corner turn to feed `mbheimdall`. TCP connections are used to ensure no data loss, as the data rates are really low. `dsaX_nicdb` feeds the `BEAMCAPTURE_BLOCK_KEY` buffer of size 64 beams x 4096 integrations x 1024 channels = 268,435,456 bytes. 

---

### scripts and utils

The "scripts" dir contains some useful scripts to test various aspects of the system (corr, bf, cornerturn). The "utils" dir includes functionality to generate fake data and beamforming weights. 



 
 
 

