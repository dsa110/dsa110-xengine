BINDIR=/usr/local/dsaX/bin

CC=gcc
CFLAGS = -g -O2 -pthread -I/usr/local/psrdada/src

CDEPS=dsaX_correlator_def.h dsaX_correlator_udpdb_thread.h
COBJ = dsaX_correlator_udpdb_thread.o
LIBS = -L/usr/local/lib -lpsrdada -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

dsaX_correlator_udpdb_thread.o: dsaX_correlator_udpdb_thread.c $(CDEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

dsaX_correlator_udpdb_thread: $(COBJ)
	gcc -o $@ $^ $(CFLAGS) $(LIBS)

dsaX_correlator_fanout: dsaX_correlator_fanout.cu
	/usr/local/cuda-8.0/bin/nvcc -D CUDA -o $@ $^ -ccbin=g++ -I/usr/local/thrust/thrust -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm --gpu-architecture=sm_61 -O3 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

dsaX_writevis: dsaX_writevis.c
	gcc -o $@ $^ -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -lcfitsio

dsaX_fancy: dsaX_fancy.cu
	/usr/local/cuda-8.0/bin/nvcc -D CUDA -o $@ $^ -ccbin=g++ -I/usr/local/thrust/thrust -I/usr/local/psrdada/src -I/usr/local/include -I/usr/local/xGPU/src -L/usr/local/lib -lpsrdada -lm -lcfitsio --gpu-architecture=sm_61 -O3 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -lxgpu -Xcompiler -fopenmp

dsaX_single: dsaX_single.cu
	/usr/local/cuda-8.0/bin/nvcc -D CUDA -o $@ $^ -ccbin=g++ -I/usr/local/thrust/thrust -I/usr/local/psrdada/src -I/usr/local/include -I/usr/local/xGPU/src -L/usr/local/lib -lpsrdada -lm -lcfitsio --gpu-architecture=sm_61 -O3 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -Xcompiler -fopenmp

dsaX_ftus: dsaX_ftus.cu
	/usr/local/cuda-8.0/bin/nvcc -o $@ $^ -ccbin=g++ -I/usr/local/psrdada/src -I/usr/local/include  -L/usr/local/lib -lpsrdada -lm -O3 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -lcfitsio --gpu-architecture=sm_61

dsaX_flag: dsaX_flag.cu
	/usr/local/cuda-8.0/bin/nvcc -o $@ $^ -ccbin=g++ -I/usr/local/psrdada/src -I/usr/local/include  -L/usr/local/lib -lpsrdada -lm -O3 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -lcfitsio --gpu-architecture=sm_61

dsaX_dbnic: dsaX_dbnic.c
	gcc -o $@ $^ -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

dsaX_correlator_copy: dsaX_correlator_copy.c
	g++ -o $@ $^ -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -O3 -g -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

dsaX_store: dsaX_store.c
	g++ -o $@ $^ -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -O3 -g -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

dsaX_trigger: dsaX_trigger.c
	gcc -o $@ $^ -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

dsaX_nicdb: dsaX_nicdb.c
	gcc -o $@ $^ -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

dsaX_recv_bsp: dsaX_recv_bsp.c
	gcc -o $@ $^ -I/usr/local/psrdada/src -I/usr/local/include -L/usr/local/lib -lpsrdada -lm -pthread -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran

install:
	cp dsaX_dbnic dsaX_nicdb dsaX_correlator_fanout dsaX_single dsaX_fancy dsaX_writevis dsaX_correlator_copy dsaX_store dsaX_trigger dsaX_correlator_udpdb_thread cpscr.bash $(BINDIR)


.PHONY: clean

clean:
	rm -f *.o *.~ dsaX_dbnic dsaX_nicdb dsaX_correlator_fanout dsaX_final dsaX_massager dsaX_xgpu dsaX_single dsaX_writevis dsaX_fancy dsaX_correlator_udpdb_thread dsaX_correlator_copy dsaX_store dsaX_trigger



