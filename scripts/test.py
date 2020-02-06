import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dsaX_testVec',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_xgpu',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_writevis',shell=True)
    subprocess.Popen.wait(output0)
    

    # destroy buffers
    os.system('dada_db -k dada -d')
    os.system('dada_db -k fada -d')
    os.system('dada_db -k aada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes
     
    # CAPTURE
    os.system('dada_db -k dada -b 75497472 -l -p -c 0 -n 8')
    # XGPU
    os.system('dada_db -k fada -b 3342336 -l -p -c 0 -n 4')
    # TEST
    os.system('dada_db -k aada -b 75497472 -l -p -c 0 -n 8')
    
    # start code    
    xgpu = '/home/dsa/dsa110-xengine/src/dsaX_xgpu -c 2'
    write = '/home/dsa/dsa110-xengine/src/dsaX_writevis -c 3 -i 192.168.3.29'
    junk = 'dada_junkdb -t 1000 -k aada -r 70 /home/dsa/dsa110-xengine/src/correlator_header_dsaX.txt'
    test = '/home/dsa/dsa110-xengine/src/dsaX_testVec -c 1'
    
        
    print('Starting write')
    write_log = open('/home/dsa/tmp/write.log','w')
    write_proc = subprocess.Popen(write, shell = True, stdout=write_log, stderr=write_log)
    sleep(0.1)    
    
    print('Starting xgpu')
    xgpu_log = open('/home/dsa/tmp/xgpu.log','w')
    xgpu_proc = subprocess.Popen(xgpu, shell = True, stdout=xgpu_log, stderr=xgpu_log)
    sleep(0.1)

    print('Starting testVec')
    test_log = open('/home/dsa/tmp/test.log','w')
    test_proc = subprocess.Popen(test, shell = True, stdout=test_log, stderr=test_log)
    sleep(0.1)

    print('Starting junkdb')
    junk_log = open('/home/dsa/tmp/junk.log','w')
    junk_proc = subprocess.Popen(junk, shell = True, stdout=junk_log, stderr=junk_log)
    sleep(0.1)

    
    
    
