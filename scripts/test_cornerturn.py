import sys, subprocess, os
from time import sleep

'''

run 2x1 cornerturn. input buffers are 64x128x48=393216, output are 268435456

'''

if sys.argv[1]=='stop':

    output0 = subprocess.Popen('killall -q dada_dbnull',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_nicdb',shell=True)
    subprocess.Popen.wait(output0)

    output0 = subprocess.Popen('killall -q dsaX_dbnic',shell=True)
    subprocess.Popen.wait(output0)
    
    output0 = subprocess.Popen('killall -q dada_junkdb',shell=True)
    subprocess.Popen.wait(output0)

    # destroy buffers
    os.system('dada_db -k bcda -d')
    os.system('dada_db -k dcda -d')
    os.system('dada_db -k bbda -d')
    
if sys.argv[1]=='start':

    # create buffers - check dsaX_def for correct block sizes

    # TEST DATA
    os.system('dada_db -k dcda -b 393216 -l -p -c 0 -n 4') 
    # CAPTURE
    os.system('dada_db -k bcda -b 393216 -l -p -c 0 -n 4') 
    # REORDER
    os.system('dada_db -k bbda -b 268435456 -l -p -c 0 -n 4') 
    
    # start code    
    junk1 = 'dada_junkdb -t 1000 -k dcda -r 3 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'
    junk2 = 'dada_junkdb -t 1000 -k bcda -r 3 /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt'
    dbnull = 'dada_dbnull -k bbda'
    dbnic1 = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_dbnic -g 0 -d -i dcda -w 127.0.0.1'
    dbnic2 = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_dbnic -g 1 -d -i bcda -w 127.0.0.1'
    nicdb = '/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_nicdb -d -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt -c 0 -o bbda -i 127.0.0.1'
    
    print('Starting dbnull')
    dbnull1_log = open('/home/ubuntu/tmp/dbnull1.log','w')
    dbnull1_proc = subprocess.Popen(dbnull, shell = True, stdout=dbnull1_log, stderr=dbnull1_log)
    sleep(0.1)
    
    print('Starting nicdb')
    nicdb_log = open('/home/ubuntu/tmp/nicdb.log','w')
    nicdb_proc = subprocess.Popen(nicdb, shell = True, stdout=nicdb_log, stderr=nicdb_log)
    sleep(0.1)
    
    print('Starting dbnic1')
    dbnic1_log = open('/home/ubuntu/tmp/dbnic1.log','w')
    dbnic1_proc = subprocess.Popen(dbnic1, shell = True, stdout=dbnic1_log, stderr=dbnic1_log)
    sleep(0.1)

    print('Starting dbnic2')
    dbnic2_log = open('/home/ubuntu/tmp/dbnic2.log','w')
    dbnic2_proc = subprocess.Popen(dbnic2, shell = True, stdout=dbnic2_log, stderr=dbnic2_log)
    sleep(0.1)

    print('Starting capture1')
    capture1_log = open('/home/ubuntu/tmp/capture1.log','w')
    capture1_proc = subprocess.Popen(junk1, shell = True, stdout=capture1_log, stderr=capture1_log)
    sleep(0.01)

    print('Starting capture2')
    capture2_log = open('/home/ubuntu/tmp/capture2.log','w')
    capture2_proc = subprocess.Popen(junk2, shell = True, stdout=capture2_log, stderr=capture2_log)

    
    
    
    
