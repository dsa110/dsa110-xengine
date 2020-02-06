BINDIR=/usr/local/dsaX/bin

CC=gcc
CFLAGS1 = -g -O2 -pthread -I/usr/local/psrdada/src -I/usr/local/include
CDEPS1=dsaX_def.h dsaX_capture.h
LIBS = -L/usr/local/lib -lpsrdada -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran  -L/usr/local/cuda/lib -lcudart -lm -lxgpu -lcfitsio

CCU=/usr/local/cuda/bin/nvcc -D CUDA -ccbin=g++
CFLAGS2 = -I/home/dsa/dsa110-xengine/src -I/usr/local/xGPU/src -I/usr/local/thrust/thrust -I/usr/local/psrdada/src -I/usr/local/include --gpu-architecture=sm_75 -O3 

dsaX_capture.o: dsaX_capture.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_capture: dsaX_capture.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_writevis.o: dsaX_writevis.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_writevis: dsaX_writevis.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

dsaX_reorder_raw: dsaX_reorder_raw.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_testVec: dsaX_testVec.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_xgpu: dsaX_xgpu.cu
	$(CCU) -o $@ $^ $(CFLAGS2) $(LIBS)


.PHONY: clean

clean:
	rm -f *.o *.~ 


