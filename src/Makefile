# This is set up for the CORR containers

CC=gcc
CFLAGS1 = -g -O3 -Wall -pthread -march=native -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include/ -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc
CDEPS1=dsaX_def.h dsaX_capture_manythread.h
CDEPS2=dsaX_def.h dsaX_capture.h
LIBS = -L/usr/local/lib -lpsrdada -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -L/usr/local/cfitsio-3.47/lib -lcfitsio -lsigproc -lxgpu

#LIBS2 = -L/home/ubuntu/PF_RING/userland/libpcap-1.9.1 -lpcap
#CDEPS3=dsaX_def.h dsaX_capture_pcap.h

CCU=/usr/local/cuda/bin/nvcc -D CUDA -ccbin=g++
CFLAGS2 = -I/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src -I/home/ubuntu/proj/dsa110-shell/dsa110-xGPU/src -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc -arch=sm_75 -O3 -Xcompiler="-pthread" -DMATRIX_ORDER_TRIANGULAR -std=c++14


.DEFAULT_GOAL := all

test_write.o: test_write.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

test_write: test_write.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

test_read.o: test_read.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

test_read: test_read.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_trigger.o: dsaX_trigger.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_trigger: dsaX_trigger.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_filTrigger.o: dsaX_filTrigger.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_filTrigger: dsaX_filTrigger.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

splice_offline_beams.o: splice_offline_beams.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

splice_offline_beams: splice_offline_beams.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_store.o: dsaX_store.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_store: dsaX_store.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_fluff.o: dsaX_fluff.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_fluff: dsaX_fluff.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_reorder.o: dsaX_reorder.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_reorder: dsaX_reorder.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)


dsaX_dbnic.o: dsaX_dbnic.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_dbnic: dsaX_dbnic.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_nicdb.o: dsaX_nicdb.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_nicdb: dsaX_nicdb.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_capture.o: dsaX_capture.c $(CDEPS2)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_capture: dsaX_capture.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_capture_thread.o: dsaX_capture_thread.c $(CDEPS2)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_capture_thread: dsaX_capture_thread.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_capture_manythread.o: dsaX_capture_manythread.c $(CDEPS2)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_capture_manythread: dsaX_capture_manythread.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_split.o: dsaX_split.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_split: dsaX_split.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_merge.o: dsaX_merge.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_merge: dsaX_merge.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_simplesplit.o: dsaX_simplesplit.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_simplesplit: dsaX_simplesplit.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)


dsaX_fake.o: dsaX_fake.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_fake: dsaX_fake.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_splitup.o: dsaX_splitup.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_splitup: dsaX_splitup.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_copydb.o: dsaX_copydb.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_copydb: dsaX_copydb.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_writevis.o: dsaX_writevis.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_writevis: dsaX_writevis.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_wrangle.o: dsaX_wrangle.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_wrangle: dsaX_wrangle.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_testdada.o: dsaX_testdada.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_testdada: dsaX_testdada.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_writeFil.o: dsaX_writeFil.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_splice.o: dsaX_splice.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_writeFil: dsaX_writeFil.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_reorder_raw.o: dsaX_reorder_raw.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_reorder_raw: dsaX_reorder_raw.o $(CDEPS1)
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

fil2dada.o: fil2dada.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

fil2dada: fil2dada.o $(CDEPS1)
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dumpfil.o: dumpfil.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dumpfil: dumpfil.o $(CDEPS1)
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_xgpu: dsaX_xgpu.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

cuda_correlator: cuda_correlator.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

gpu_flagger: gpu_flagger.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_beamformer: dsaX_beamformer.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_bfCorr: dsaX_bfCorr.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_beamformer_passon: dsaX_beamformer_passon.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_beamformer_offline: dsaX_beamformer_offline.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

.PHONY: clean all

clean:
	rm -f *.o *~ dsaX_beamformer dsaX_beamformer_passon dsaX_xgpu dsaX_reorder_raw dsaX_writeFil dsaX_writevis dsaX_fake dsaX_capture dsaX_dbnic dsaX_nicdb dsaX_split dsaX_wrangle fil2dada gpu_flagger dumpfil dsaX_simplesplit dsaX_store dsaX_trigger dsaX_beamformer_offline dsaX_splice dsaX_filTrigger cuda_correlator dsaX_copydb dsaX_bfCorr dsaX_merge

all: dsaX_beamformer dsaX_beamformer_passon dsaX_xgpu dsaX_reorder_raw dsaX_writeFil dsaX_writevis dsaX_fake dsaX_capture dsaX_capture_thread dsaX_capture_manythread dsaX_dbnic dsaX_nicdb dsaX_split dsaX_wrangle fil2dada gpu_flagger dumpfil dsaX_simplesplit dsaX_store dsaX_trigger dsaX_filTrigger dsaX_beamformer_offline dsaX_splice dsaX_splitup cuda_correlator dsaX_copydb dsaX_bfCorr dsaX_merge




