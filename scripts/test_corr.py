import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dada_dbnull',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_xgpu',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dsaX_reorder_raw',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_writevis',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_fake',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k dada -d')
    os.system('dada_db -k eada -d')
    os.system('dada_db -k fada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k aada -b 198180864 -l -p -c 0 -n 4') # 2048 packets
    # CAPTURE
    os.system('dada_db -k dada -b 198180864 -l -p -c 0 -n 8') # 2048 packets
    # REORDER
    os.system('dada_db -k eada -b 402653184 -l -p -c 0 -n 8') # 2048 packets
    # XGPU
    #os.system('dada_db -k fada -b 51904512 -l -p -c 0 -n 4') # output order (register tile)
    os.system('dada_db -k fada -b 51118080 -l -p -c 0 -n 4') # output order (triangular)

    
    # start code    
    junk = 'dada_junkdb -t 1000 -k aada -r 1500 /home/dsa/dsa110-xengine/src/correlator_header_dsaX.txt'
    reorder = '/home/dsa/dsa110-run/src/dsaX_reorder_raw -t 16'
    xgpu = '/home/dsa/dsa110-run/src/dsaX_xgpu -t 4'
    writevis = '/home/dsa/dsa110-run/src/dsaX_writevis -i 10.40.0.23 -f /home/dsa/data/testing -c 15'
    dbnull = 'dada_dbnull -k fada'
    fake = '/home/dsa/dsa110-run/src/dsaX_fake -f /home/dsa/dsa110-run/utils/packet.out'

    
    #print('Starting dbnull')
    #dbnull_log = open('/home/dsa/tmp/dbnull.log','w')
    #dbnull_proc = subprocess.Popen(dbnull, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    #sleep(0.1)

    print('Starting writevis')
    writevis_log = open('/home/dsa/tmp/writevis.log','w')
    writevis_proc = subprocess.Popen(writevis, shell = True, stdout=writevis_log, stderr=writevis_log)
    sleep(0.1)
    
    
    print('Starting xgpu')
    xgpu_log = open('/home/dsa/tmp/xgpu.log','w')
    xgpu_proc = subprocess.Popen(xgpu, shell = True, stdout=xgpu_log, stderr=xgpu_log)
    sleep(0.1)
    
    print('Starting reorder')
    reorder_log = open('/home/dsa/tmp/reorder.log','w')
    reorder_proc = subprocess.Popen(reorder, shell = True, stdout=reorder_log, stderr=reorder_log)
    sleep(0.1)

    print('Starting fake')
    fake_log = open('/home/dsa/tmp/fake.log','w')
    fake_proc = subprocess.Popen(fake, shell = True, stdout=fake_log, stderr=fake_log)
    sleep(0.1)
    
    print('Starting capture')
    capture_log = open('/home/dsa/tmp/capture.log','w')
    capture_proc = subprocess.Popen(junk, shell = True, stdout=capture_log, stderr=capture_log)
    sleep(0.1)

    
    
    
