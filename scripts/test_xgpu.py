import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dada_dbnull',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_xgpu',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    # destroy buffers
    os.system('dada_db -k dada -d')
    os.system('dada_db -k bada -d')
    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # CAPTURE
    os.system('dada_db -k dada -b 201326592 -l -p -c 0 -n 20') # 2048 packets
    # REORDER
    os.system('dada_db -k bada -b 51118080 -l -p -c 0 -n 8') # 2048 packets

    
    # start code    
    junk = 'dada_junkdb -t 1000 -k dada -r 1600 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'
    reorder = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_xgpu -i dada -o bada -d -c 6'
    dbnull1 = 'dada_dbnull -k bada'

    
    print('Starting dbnull1')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(dbnull1, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting reorder')
    reorder_log = open('/home/ubuntu/tmp/reorder.log','w')
    reorder_proc = subprocess.Popen(reorder, shell = True, stdout=reorder_log, stderr=reorder_log)
    sleep(2)
    
    print('Starting capture')
    capture_log = open('/home/ubuntu/tmp/capture.log','w')
    capture_proc = subprocess.Popen(junk, shell = True, stdout=capture_log, stderr=capture_log)
    sleep(0.1)

    
    
    
