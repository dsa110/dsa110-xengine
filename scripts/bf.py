import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dsaX_beamformer',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_fake',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dada_dbnull',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')
    os.system('dada_db -k cada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k aada -b 94371840 -l -p -c 1 -n 8') # 2048 packets, 30 ants
    os.system('dada_db -k bada -b 94371840 -l -p -c 1 -n 8') # 2048 packets, 30 ants
    os.system('dada_db -k cada -b 6291456 -l -p -c 1 -n 8') # 256 beams, 512 ints, 48 chans

    
    
    # start code    
    junk = 'dada_junkdb -t 1000 -k aada -r 704 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'    
    dbnull = 'dada_dbnull -k cada'
    fake = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_fake -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/packet.out -i aada -o bada'
    bf = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_beamformer -c 30 -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out -i bada -o cada -z 1450.0 -a /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat'
    
    
    print('Starting dbnull')
    dlog = open('/home/ubuntu/tmp/dbnull.log','w')
    proc = subprocess.Popen(dbnull, shell = True, stdout=dlog, stderr=dlog)
    sleep(0.1)

    print('Starting beamformer')
    dlog = open('/home/ubuntu/tmp/dbnull.log','w')
    proc = subprocess.Popen(bf, shell = True, stdout=dlog, stderr=dlog)
    sleep(0.1)

    print('Starting fake')
    dlog = open('/home/ubuntu/tmp/dbnull.log','w')
    proc = subprocess.Popen(fake, shell = True, stdout=dlog, stderr=dlog)
    sleep(0.1)

    print('Starting junk')
    dlog = open('/home/ubuntu/tmp/dbnull.log','w')
    proc = subprocess.Popen(junk, shell = True, stdout=dlog, stderr=dlog)
    sleep(0.1)

    
    
    
