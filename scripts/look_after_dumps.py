#!/usr/bin/env python
"""script to control and monitor the correlator
   
"""

import argparse
import json
from time import sleep
import time
import yaml
from os.path import dirname
from os.path import realpath
import glob
import sys
import subprocess
import os
import socket
import numpy as np
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
my_log = dsl.DsaSyslogger()
my_log.subsystem('correlator')
my_log.app('look_after_dumps.py')
from astropy.time import Time
from os import path
import re

        
# watch callback function for commands
def cb_func(my_ds):
    """ etcd watch callback function
    """

    def a(event):

        my_log.function('cb_func')        
        my_log.debug("received event= {}".format(event))
        print(event)
        tm=list(event)[0]
        #tm = (int(list(event)[0])-1907)*4
        my_log.info("name = {}".format(tm))
        with open('/home/ubuntu/data/'+tm+'.json', 'w') as f: #encoding='utf-8'            
            json.dump(event, f, ensure_ascii=False, indent=4)
        
    return a
                                                                                                


def ld_run(args):

    """Main entry point. Will never return.
    :param args: Input arguments from argparse.
    """
    # open logger
    my_log.function('ld_run')

    # parse argument
    my_log.debug('corr node num '+str(args.corr_num))

    # connect to etcd
    my_ds = ds.DsaStore()

    # callback on triggers
    my_ds.add_watch('/mon/corr/1/trigger', cb_func(my_ds))
    
    # infinite monitoring loop
    while True:

        # test for existence of file in data dir
        if len(glob.glob('/home/ubuntu/data/fl_0.out*'))>0:

            # find latest out file that hasn't been moved
            lf = glob.glob('/home/ubuntu/data/fl_*.out')
            if len(lf)>0:

                llf = max(lf,key=path.getctime)
            
                # extract fl number
                flnum = int(re.findall('[0-9]+', llf)[0])

                # find specnum number
                os.system("grep specnum /home/ubuntu/data/dumps.dat | awk '{print $6,$7,$8}' | sed 's/NUM/ /' | sed 's/NUM/ /' | awk '{print $5,$4}' > /home/ubuntu/tmp/specnums.dat")
                specnum,dumpnum = np.genfromtxt("/home/ubuntu/tmp/specnums.dat",dtype='str').transpose()
                dumpnum=dumpnum.astype('int')
                cur_specnum = specnum[dumpnum==flnum][0]

                # test for existence of associated json file
                if path.exists(llf+"."+str(cur_specnum)+".json"):
                    sleep(1)

                else:
                
                    # simply copy associated json file, and copy llf file
                    os.system("cp /home/ubuntu/data/"+str(cur_specnum)+".json "+llf+"."+str(cur_specnum)+".json")
                    os.system("mv "+llf+" "+llf+"."+str(cur_specnum))

                    sleep(1)

        else:

            sleep(1)

    

                
                
                
                
            
                                                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--corr_num', type=int, default='1', help='corr node number')
    the_args = parser.parse_args()
    ld_run(the_args)

    
    
