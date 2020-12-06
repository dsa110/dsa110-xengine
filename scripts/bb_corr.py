import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dumpfil',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_wrangle',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dsaX_xgpu',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_reorder_raw',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q fil2dada',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')
    os.system('dada_db -k cada -d')
    os.system('dada_db -k dada -d')
    os.system('dada_db -k eada -d')

    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k aada -b 75497472 -l -p -c 1 -n 8') # 2048 packets, 24 ants
    os.system('dada_db -k bada -b 75497472 -l -p -c 1 -n 8') # 2048 packets, 24 ants
    os.system('dada_db -k cada -b 402653184 -l -p -c 1 -n 4') # 2048 packets, 24 ants
    os.system('dada_db -k dada -b 51118080 -l -p -c 1 -n 4') # 2048 packets, 24 ants
    os.system('dada_db -k eada -b 1996800 -l -p -c 1 -n 4') # 2048 packets, 24 ants
    
    
    # start code    
    junk = 'dada_junkdb -t 1000 -k aada -r 200 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'    
    fake = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/fil2dada -f /home/ubuntu/data/fl_corr06.out -i aada -o bada -n'
    reorder = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_reorder_raw -t 16 -i bada -o cada'
    corr = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_xgpu -t 8 -i cada -o dada -c 31'
    wrangle = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_wrangle -i dada -o eada'
    dump = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dumpfil -f /home/ubuntu/tmp/test.out -i eada -p'
    
    
    print('Starting dump')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(dump, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting wrangle')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(wrangle, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting corr')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(corr, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting reorder')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(reorder, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)
    
    print('Starting fake')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(fake, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)

    print('Starting junk')
    dbnull_log = open('/home/ubuntu/tmp/log.log','w')
    dbnull_proc = subprocess.Popen(junk, shell = True, stdout=dbnull_log, stderr=dbnull_log)
    sleep(0.1)
    
    
    
