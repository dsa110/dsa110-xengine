import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dsaX_beamformer',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_fake',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_dbnull',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_xgpu',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_reorder_raw',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')    
    os.system('dada_db -k cada -d')
    os.system('dada_db -k dada -d')
    os.system('dada_db -k eada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    os.system('dada_db -k aada -b 198180864 -l -p -c 1 -n 8') # 2048 packets
    os.system('dada_db -k bada -b 198180864 -l -p -c 1 -n 30 -r 2') # 2048 packets
    os.system('dada_db -k cada -b 201326592 -l -p -c 1 -n 20') # 2048 packets
    os.system('dada_db -k dada -b 51118080 -l -p -c 0 -n 8') # 2048 packets
    os.system('dada_db -k eada -b 6291456 -l -p -c 1 -n 1') # 2048 packets, 63 ants
    
    # start code    
    junk = 'dada_junkdb -t 3600 -k aada -r 1500 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'
    fake = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_fake -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/packet.out -i aada -o bada'
    reorder = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_reorder_raw -t 16 -i bada -o cada'
    bf = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_beamformer -c 30 -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out -i bada -o eada -z 1450.0 -a /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -q'
    xgpu = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_xgpu -i cada -o dada -d -c 6'
    dbnull1 = 'dada_dbnull -k dada'
    dbnull2 = 'dada_dbnull -k eada'
    
    
    
    print('Starting dbnull1')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(dbnull1, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting dbnull2')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(dbnull2, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting xgpu')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(xgpu, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting reorder')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(reorder, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting bf')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(bf, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting fake')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(fake, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting junk')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(junk, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    
    
    
    
    
    
