import sys, os

if sys.argv[1]=='capture':
    os.system('echo -n '+sys.argv[2]+'-'+sys.argv[3]+' | nc -4u -w1 192.168.3.29 11223')

if sys.argv[1]=='writevis':
    os.system('echo -n '+sys.argv[2]+'-'+sys.argv[3]+'- | nc -4u -w1 192.168.3.29 11226 &')
