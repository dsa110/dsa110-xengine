import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dsaX_writeFil',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dsaX_beamformer',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_fake',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_split',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')
    #os.system('dada_db -k cada -d')
    os.system('dada_db -k dada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k aada -b 75497472 -l -p -c 1 -n 8') # 2048 packets, 24 ants
    os.system('dada_db -k bada -b 75497472 -l -p -c 1 -n 8') # 2048 packets, 24 ants
    #os.system('dada_db -k cada -b 75497472 -l -p -c 1 -n 8') # 2048 packets, 24 ants
    os.system('dada_db -k dada -b 1572864 -l -p -c 1 -n 8') # 2048 packets, 24 ants

    
    
    # start code    
    junk = 'dada_junkdb -t 1000 -k aada -r 563 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'    
    #dbnull = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_writeFil -c 27 -i 127.0.0.1 -f /home/ubuntu/data/bsfil -k dada'
    dbnull = 'dada_dbnull -k dada'
    fake = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_fake -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/packet.out -i aada -o bada'
    bf = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_beamformer -c 30 -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out -i bada -o dada'
    #split = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_split -c 18 -m -i bada -o cada'
    
    
    print('Starting dbnull')
    dbnull_log = open('/home/ubuntu/tmp/dbnull.log','w')
    dbnull_proc = subprocess.Popen(dbnull, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting bf')
    bf_log = open('/home/ubuntu/tmp/bf.log','w')
    bf_proc = subprocess.Popen(bf, shell = True, stdout=bf_log, stderr=bf_log)
    sleep(0.1)

    #print('Starting split')
    #bf_log = open('/home/ubuntu/tmp/split.log','w')
    #bf_proc = subprocess.Popen(split, shell = True, stdout=bf_log, stderr=bf_log)
    #sleep(0.1)
    
    print('Starting fake')
    fake_log = open('/home/ubuntu/tmp/fake.log','w')
    fake_proc = subprocess.Popen(fake, shell = True, stdout=fake_log, stderr=fake_log)
    sleep(0.1)
    
    print('Starting junk')
    capture_log = open('/home/ubuntu/tmp/capture.log','w')
    capture_proc = subprocess.Popen(junk, shell = True, stdout=capture_log, stderr=capture_log)
    sleep(0.1)

    
    
    
