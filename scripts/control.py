import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dsaX_capture',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_xgpu',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_writevis',shell=True)
    subprocess.Popen.wait(output0)

    # destroy buffers
    os.system('dada_db -k dada -d')
    os.system('dada_db -k fada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes
     
    # CAPTURE
    os.system('dada_db -k dada -b 75497472 -l -p -c 0 -n 8')
    # XGPU
    os.system('dada_db -k fada -b 3342336 -l -p -c 0 -n 4')

    # start code
    capture = '/home/dsa/dsa110-xengine/src/dsaX_capture -c 1 -j 10.41.0.43 -i 192.168.3.29 -f /home/dsa/dsa110-xengine/src/correlator_header_dsaX.txt'
    xgpu = '/home/dsa/dsa110-xengine/src/dsaX_xgpu -c 2'
    dbnull = '/home/dsa/dsa110-xengine/src/dsaX_writevis -c 3 -i 192.168.3.29'

        
    print('Starting dbnull')
    dbnull_log = open('/home/dsa/tmp/dbnull.log','w')
    dbnull_proc = subprocess.Popen(dbnull, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)    
    
    print('Starting xgpu')
    xgpu_log = open('/home/dsa/tmp/xgpu.log','w')
    xgpu_proc = subprocess.Popen(xgpu, shell = True, stdout=xgpu_log, stderr=xgpu_log)
    sleep(0.1)

    print('Starting capture')
    capture_log = open('/home/dsa/tmp/capture.log','w')
    capture_proc = subprocess.Popen(capture, shell = True, stdout=capture_log, stderr=capture_log)
    sleep(0.1)
    
    
