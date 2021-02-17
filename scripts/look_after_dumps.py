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
my_log.app('corr.py')
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
        tm = (int(list(event)[0])-477)*16
        my_log.info("specnum = {}".format(tm))
        with open('/home/user/ubuntu/'+str(tm)+'.json', 'w') as f: #encoding='utf-8'            
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
        if path.exists('/home/ubuntu/data/fl_0.out'):

            # find latest out file
            lf = glob.glob('/home/ubuntu/data/fl_*.out')
            llf = max(lf,key=path.getctime)

            # test for existence of associated json file
            if path.exists(llf+'.json'):
                sleep(1)

            else:

                # extract fl number
                flnum = int(re.findall('[0-9]+', llf)[0])

                # find specnum number
                os.system("grep specnum /home/ubuntu/data/dumps.dat | awk '{print $5,$6,$7}' | sed 's/NUM/ /' | sed 's/NUM/ /' | awk '{print $1,$5}' > /home/ubuntu/tmp/specnums.dat")
                specnum,dumpnum = np.loadtxt("/home/ubuntu/tmp/specnums.dat").transpose()
                cur_specnum = specnum[dumpnum==flnum]

                # simply copy associated json file
                os.sysem("cp /home/ubuntu/data/"+str(cur_specnum)+".json /home/ubuntu/data/"+llf+".json")

                sleep(1)

        else:

            sleep(1)

    

                
                
                
                
            
                                                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--corr_num', type=int, default='1', help='corr node number')
    the_args = parser.parse_args()
    ld_run(the_args)

    
    
