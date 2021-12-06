import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dsaX_writeFil',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q gpu_flagger',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')
    
if sys.argv[1]=='create':

    # create buffers - check dsaX_def for correct block sizes

    os.system('dada_db -k aada -b 1073741824 -c 0 -n 4') # 2048 packets
    os.system('dada_db -k bada -b 1073741824 -c 0 -n 4') # 2048 packets

if sys.argv[1]=='start':
    
    # start code    
    junk = 'dada_junkdb -t 1200 -k aada -r 200 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/spectrometer_header.txt'
    flag = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/gpu_flagger -i aada -o bada -t 4.5 -d -f /home/ubuntu/tmp/test.dat -v -n -k 127.0.0.1 -m'
    fil = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_writeFil -f /home/ubuntu/data/testAdd -k bada -i 127.0.0.1'
    
    print('Starting fil')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(fil, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting flagger')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(flag, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting junkdb')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(junk, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    
