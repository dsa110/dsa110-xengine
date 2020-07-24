BINDIR=/usr/local/dsaX/bin

CC=gcc
CFLAGS1 = -g -O3 -Wall -pthread -march=native -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include/
CDEPS1=dsaX_def.h dsaX_capture.h
LIBS = -L/usr/local/lib -lpsrdada -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -L/usr/local/cuda-11.0/targets/x86_64-linux/lib -lcudart -lm -lxgpu -L/usr/local/cfitsio-3.47/lib -lcfitsio -lsigproc

CCU=/usr/local/cuda/bin/nvcc -D CUDA -ccbin=g++
CFLAGS2 = -I/home/dsa/dsa110-xengine/src -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include -arch=sm_75 -O3 -Xcompiler="-pthread" -DMATRIX_ORDER_TRIANGULAR
#CFLAGS2 = -I/home/dsa/dsa110-xengine/src -I/usr/local/psrdada/src -I/usr/local/include -arch=sm_75 -O3

.DEFAULT_GOAL := all

dsaX_dbnic.o: dsaX_dbnic.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_dbnic: dsaX_dbnic.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_nicdb.o: dsaX_nicdb.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_nicdb: dsaX_nicdb.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_capture.o: dsaX_capture.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_capture: dsaX_capture.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_split.o: dsaX_split.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_split: dsaX_split.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_fake.o: dsaX_fake.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_fake: dsaX_fake.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_writevis.o: dsaX_writevis.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_writevis: dsaX_writevis.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_writeFil.o: dsaX_writeFil.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_writeFil: dsaX_writeFil.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_reorder_raw.o: dsaX_reorder_raw.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_reorder_raw: dsaX_reorder_raw.o $(CDEPS1)
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_xgpu: dsaX_xgpu.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_beamformer: dsaX_beamformer.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

.PHONY: clean all

clean:
	rm -f *.o *~ 

all: dsaX_beamformer dsaX_xgpu dsaX_reorder_raw dsaX_writeFil dsaX_writevis dsaX_fake dsaX_capture dsaX_dbnic dsaX_nicdb


