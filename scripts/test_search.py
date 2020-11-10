import sys, subprocess, os
from time import sleep

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q heimdall',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dumpfil',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q gpu_flagger',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q fil2dada',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)
    
    # destroy buffers
    os.system('dada_db -k aada -d')
    os.system('dada_db -k bada -d')
    os.system('dada_db -k cada -d')
    
if sys.argv[1]=='create':

    # create buffers - check dsaX_def for correct block sizes

    os.system('dada_db -k aada -b 268435456 -l -p -c 0 -n 4') # 2048 packets
    os.system('dada_db -k bada -b 268435456 -l -p -c 0 -n 4') # 2048 packets
    os.system('dada_db -k cada -b 268435456 -l -p -c 0 -n 4') # 2048 packets

if sys.argv[1]=='start':
    
    # start code    
    junk = 'dada_junkdb -t 1200 -k aada -r 67 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/spectrometer_header.txt'
    fil = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/fil2dada -f /home/ubuntu/data/bsfil_B0531_1.fil -i aada -o bada'
    flag = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/gpu_flagger -i bada -o cada -t 4.5 -d -f /home/ubuntu/tmp/test.dat -v -n'
    heimdall = '/home/ubuntu/proj/dsa110-shell/dsa110-mbheimdall/bin/heimdall -k cada -gpu_id 0 -nsamps_gulp 4096 -output_dir /home/ubuntu/data -dm 10 1000 -dm_tol 1.35 -max_giant_rate 20000000 -nbeams 64'
    #heimdall = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dumpfil -n 3 -f /home/ubuntu/data/test.fil -i cada -g'
    
    print('Starting heimdall')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(heimdall, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting flagger')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(flag, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting fil2dada')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(fil, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    print('Starting junkdb')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(junk, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)

    
