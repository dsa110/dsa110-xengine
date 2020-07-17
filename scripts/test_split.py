import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dada_dbnull',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_split',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    # destroy buffers
    os.system('dada_db -k dada -d')
    os.system('dada_db -k abda -d')
    os.system('dada_db -k bada -d')
    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k abda -b 198180864 -l -p -c 0 -n 4') # 2048 packets
    # CAPTURE
    os.system('dada_db -k dada -b 198180864 -l -p -c 0 -n 8') # 2048 packets
    # REORDER
    os.system('dada_db -k bada -b 198180864 -l -p -c 0 -n 8') # 2048 packets

    
    # start code    
    junk = 'dada_junkdb -t 1000 -k dada -r 1500 /home/dsa/dsa110-xengine/src/correlator_header_dsaX.txt'
    reorder = '/home/dsa/dsa110-run/src/dsaX_split -t 16 -b -d'
    dbnull1 = 'dada_dbnull -k abda'
    dbnull2 = 'dada_dbnull -k bada'

    
    print('Starting dbnull1')
    dbnull1_log = open('/home/dsa/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(dbnull1, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting dbnull2')
    dbnull2_log = open('/home/dsa/tmp/dbnull2.log','w')
    dbnull2_proc = subprocess.Popen(dbnull2, shell = True, stdout=dbnull2_log, stderr=dbnull2_log)
    sleep(0.1)
    
    print('Starting reorder')
    reorder_log = open('/home/dsa/tmp/reorder.log','w')
    reorder_proc = subprocess.Popen(reorder, shell = True, stdout=reorder_log, stderr=reorder_log)
    sleep(0.1)
    
    print('Starting capture')
    capture_log = open('/home/dsa/tmp/capture.log','w')
    capture_proc = subprocess.Popen(junk, shell = True, stdout=capture_log, stderr=capture_log)
    sleep(0.1)

    
    
    
