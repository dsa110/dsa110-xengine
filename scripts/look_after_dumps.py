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
import dsacalib.constants as ct
my_log = dsl.DsaSyslogger()
my_log.subsystem('correlator')
my_log.app('look_after_dumps.py')
from astropy.time import Time
from os import path
import re

def get_mjd(armed_mjd, utc_start, specnum):
    """Get the start mjd of a voltage dump.

    Parameters
    ----------
    armed_mjd : float
        The time at which the snaps were armed, in mjd.
    utc_start : int
        The spectrum number at which the correlator was started.
    specnum : int
        The spectrum number of the first spectrum in the voltage dump,
        referenced to when the correlator was started.

    Returns
    -------
    tstart : float
        The start time of the voltage dump in mjd.
    """
    tstart = (armed_mjd+utc_start*4*8.192e-6/86400+
              (1/(250e6/8192/2)*specnum/ct.SECONDS_PER_DAY))
    return tstart

        
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
                    jsonfile = '/home/ubuntu/data/{0}.json'.format(cur_specnum)
                    if not path.exists(jsonfile):
                        # create a dummy json file
                        with open('/home/ubuntu/data/dumps.dat', 'r') as dumpsf:
                            for line in dumpsf:
                                if re.search(cur_specnum, line):
                                    actual_specnum = int(line.split()[4])
                                    json_dictionary = dict({
                                        cur_specnum: {
                                            "mjds": get_mjd(
                                                float(my_ds.get_dict('/mon/snap/1')['armed_mjd']),
                                                int(my_ds.get_dict('/mon/snap/1/utc_start')['utc_start']),
                                                actual_specnum
                                                ),
                                            "specnum": actual_specnum,
                                            "snr": 0,
                                            "ibox": 0,
                                            "dm": 0.,
                                            "ibeam": 0,
                                            "cntb": 0,
                                            "cntc": 0
                                        }
                                    })
                                    with open(jsonfile, 'w') as jsonfhandler:
                                        json.dump(json_dictionary, jsonfhandler)
                                    break
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

    
    
