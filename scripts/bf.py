import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dada_dbnull',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_writeFil',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dsaX_beamformer',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_fake',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')
    os.system('dada_db -k dcda -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k aada -b 198180864 -l -p -c 0 -n 4') # 2048 packets, 63 ants
    #os.system('dada_db -k aada -b 84934656 -l -p -c 0 -n 4') # 2048 packets, 27 ants
    # CAPTURE
    os.system('dada_db -k bada -b 198180864 -l -p -c 0 -n 40') # 2048 packets, 63 ants
    #os.system('dada_db -k dbda -b 84934656 -l -p -c 0 -n 20') # 2048 packets, 27 ants
    # BF
    #os.system('dada_db -k dcda -b 1572864 -l -p -c 0 -n 4') # all beams (128*256*48)
    os.system('dada_db -k dcda -b 6144 -l -p -c 0 -n 4') # one beam
    
    # start code    
    junk = 'dada_junkdb -t 1000 -k aada -r 150 /home/dsa/dsa110-xengine_TESTING/src/correlator_header_dsaX.txt'
    dbnull = 'dada_dbnull -k dcda'
    #dbnull = '/home/dsa/dsa110-run/src/dsaX_writeFil -c 27 -i 10.40.0.23 -f /home/dsa/data/bsfil'
    fake = '/home/dsa/dsa110-xengine/src/dsaX_fake -f /home/dsa/dsa110-xengine/utils/packet.out'
    bf = '/home/dsa/dsa110-xengine/src/dsaX_beamformer -c 9 -f /home/dsa/dsa110-xengine/utils/antennas.out -i bada -o dcda'
    
    
    print('Starting dbnull')
    dbnull_log = open('/home/dsa/tmp/dbnull.log','w')
    dbnull_proc = subprocess.Popen(dbnull, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting bf')
    bf_log = open('/home/dsa/tmp/bf.log','w')
    bf_proc = subprocess.Popen(bf, shell = True, stdout=bf_log, stderr=bf_log)
    sleep(0.1)
    
    print('Starting fake')
    fake_log = open('/home/dsa/tmp/fake.log','w')
    fake_proc = subprocess.Popen(fake, shell = True, stdout=fake_log, stderr=fake_log)
    sleep(0.1)
    
    print('Starting junk')
    capture_log = open('/home/dsa/tmp/capture.log','w')
    capture_proc = subprocess.Popen(junk, shell = True, stdout=capture_log, stderr=capture_log)
    sleep(0.1)

    
    
    
