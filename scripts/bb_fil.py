import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dumpfil',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dsaX_beamformer',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q fil2dada',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')
    os.system('dada_db -k cada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k aada -b 75497472 -l -p -c 1 -n 8') # 2048 packets, 24 ants
    os.system('dada_db -k bada -b 75497472 -l -p -c 1 -n 8') # 2048 packets, 24 ants
    os.system('dada_db -k cada -b 1572864 -l -p -c 1 -n 8') # 2048 packets, 24 ants

    
    
    # start code    
    junk = 'dada_junkdb -t 1000 -k aada -r 300 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'    
    fake = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/fil2dada -f '+sys.argv[2]+' -i aada -o bada'
    bf = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_beamformer -c 30 -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/'+sys.argv[3]+' -i bada -o cada -a /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -z '+sys.argv[4]
    dump = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dumpfil -f /home/ubuntu/tmp/test.fil -i cada -n 30'
    
    
    print('Starting dump')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(dump, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting bf')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(bf, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting fake')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(fake, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting junk')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(junk, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)
    
    
    
