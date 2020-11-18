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

def time_to_mjd(t):
    """ converts time.time() to mjd                                                                
    """
    tt = time.gmtime(t)
    Y = tt.tm_year
    M = tt.tm_mon
    D = tt.tm_mday + tt.tm_hour/24. + tt.tm_min/24./60. + tt.tm_sec/24./60./60.

    A = np.floor(Y/100.)
    B = np.floor(A/4.)
    C = 2.-A-B
    E = np.floor(365.25*(Y+4716.))
    F = np.floor(30.6001*(M+1.))
    MJD = C+D+E+F-1514.5 - 2400000.5
    
    return(MJD)

def read_yaml(fname):
    """Read a YAML formatted file
    
    :param fn: YAML formatted filename"                                                                                    
    :type fn: String                                                                                                       
    :return: Dictionary on success. None on error                                                                          
    :rtype: Dictionary                                                                                                    
 
    """
    with open(fname, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            return None

def parse_value(value):
    """Parse the string in JSON format and assumed represent a dictionary.
    :param value: JSON string of the form: {"key":"value"} or {"key":number|bool}
    :type value: String                                                                                                    
    :return: Key,value dictionary                                                                                          
    :rtype: Dictionary                                                                                                     
    :raise: ValueError                                                                                                     

    """

    rtn = {}
    try:
        rtn = json.loads(value)
    except ValueError:
        my_log.error("parse_value(): JSON Decode Error. Check JSON. value= {}".format(value))
        return rtn
                    
def get_capture_stats():

    """gets capture stats to put in etcd
    """

    # open logger
    my_log.function('get_capture_stats')
    
    try:
        result = subprocess.check_output("tail -n 50000 /var/log/syslog | grep CAPSTATS | tail -n 1 | awk '{print $7,$10,$13,$15}'", shell=True, stderr=subprocess.STDOUT)
        arr = result.decode("utf-8").split(' ')
        oarr = np.zeros(4)
        for i in range(4):
            oarr[i] = float(arr[i])
    except:
        #my_log.warning('buffer not accessible: '+buff)
        return -1

    return oarr.tolist()
    

def get_buf_info(buff):

    """get info on dada buffer
    :param buffer: buffer name
    :type buffer: str

    returns list of buffer prope
    """

    # open logger
    my_log.function('get_buf_info')
    
    try:
        result = subprocess.check_output('dada_dbmetric -k '+buff, shell=True, stderr=subprocess.STDOUT)
    except:
        #my_log.warning('buffer not accessible: '+buff)
        return -1
    arr = result.decode("utf-8").split(',')
    oarr = np.zeros(4)
    for i in range(4):
        oarr[i] = int(arr[i+1])

    return oarr.tolist()

# this only reads in buffer information
# TODO: add outputs from code
def get_monitor_dict(params,corr_num):
    """ prepares monitor dictionary for corr
    
    :param params: corr config params
    """
    
    mon_dict = {}

    # simply insert into flat distribution of buffers

    bct = 0
    for buff in params['buffers']:

        infoo = get_buf_info(buff['k'])
        if infoo==-1:
            return -1

        mon_dict['b'+str(bct)+'_name'] = buff['k']
        mon_dict['b'+str(bct)+'_full'] = infoo[0]
        mon_dict['b'+str(bct)+'_clear'] = infoo[1]
        mon_dict['b'+str(bct)+'_written'] = infoo[2]
        mon_dict['b'+str(bct)+'_read'] = infoo[3]

        bct += 1
        
    capstats = get_capture_stats()
    if capstats==-1:
        return -1
    mon_dict['capture_rate'] = capstats[0]
    mon_dict['drop_rate'] = capstats[1]*8.
    mon_dict['drop_count'] = capstats[2]
    mon_dict['last_seq'] = capstats[3]

    # stuff Rick wants
    mon_dict['sim'] = 'false'
    mon_dict['corr_num'] = corr_num
    mon_dict['time'] = time_to_mjd(time.time())
    
    return mon_dict

# this actually processes commands
def process(params, cmd, val, my_ds):
    """ starts and stops correlator pipeline generically according to config file
    input: params, cmd, val, DsaStore
    """

    # start up logger
    my_log.function('process')

    # to send trigger
    if cmd=='trigger':
        cmdstr = 'echo '+val+' | nc -4u -w1 127.0.0.1 11227 &'
        my_log.info('running: '+cmdstr)
        os.system(cmdstr)
        sleep(0.5)
        my_log.info('Successfully issued trigger (I think)')

    # to set UTC_START
    if cmd=='utc_start':
        cmdstr = 'echo UTC_START-'+val+' | nc -4u -w1 127.0.0.1 11223 &'
        my_log.info('running: '+cmdstr)
        os.system(cmdstr)
        #set utc_start
        try:
            my_ds.put_dict('/mon/snap/1/utc_start',{'utc_start':int(val)})
        except:
            my_log.error("Could not place utc_start into etcd")

        sleep(0.5)
        
        my_log.info('Successfully issued UTC_START (I think)')

    # to set UTC_STOP
    if cmd=='utc_stop':
        cmdstr = 'echo UTC_STOP-'+val+' | nc -4u -w1 127.0.0.1 11223 &'
        my_log.info('running: '+cmdstr)
        os.system(cmdstr)
        sleep(0.5)
        my_log.info('Successfully issued UTC_STOP (I think)')
        
        
    # start up stuff
    if cmd=='start':

        # deal with buffers
        for buff in params['buffers']:
            cmdstr = 'dada_db -k '+str(buff['k'])+' -b '+str(buff['b'])+' -n '+str(buff['n'])+' -c '+str(buff['c'])+' -l -p'
            my_log.debug('running: '+cmdstr)
            os.system(cmdstr)
            sleep(0.5)

        # deal with processes
        for rout in params['routines']:
            print(rout)
            if rout.get('hostargs') is None:
                cmdstr = rout['cmd']+' '+rout['args']
            else:
                cmdstr = rout['cmd']+' '+rout['args']+' '+rout.get('hostargs')[socket.gethostname()]
            my_log.debug('running: '+cmdstr)
            my_log.info('Starting '+rout['name'])
            log = open('/home/ubuntu/tmp/log.log','w')
            proc = subprocess.Popen(cmdstr, shell = True, stdout=log, stderr=log)
            sleep(0.5)

        #zero out utc_start
        try:
            my_ds.put_dict('/mon/snap/1/utc_start',{'utc_start':10000})
        except:
            my_log.error("Could not place utc_start into etcd")
            
        my_log.info('Successfully started (I think)')

    # stop stuff
    if cmd=='stop':

        # deal with processes
        for rout in params['routines']:
            cmdstr = 'killall -q '+rout['name']
            my_log.debug('running: '+cmdstr)
            my_log.info('Stopping '+rout['name'])
            proc = subprocess.Popen(cmdstr, shell = True)
            subprocess.Popen.wait(proc)            
            
        # deal with buffers
        for buff in params['buffers']:
            cmdstr = 'dada_db -k '+str(buff['k'])+' -d'
            my_log.debug('running: '+cmdstr)
            os.system(cmdstr)
            sleep(0.5)

        my_log.info('Successfully stopped (I think)')
        
        
# watch callback function for commands
def cb_func(params,my_ds):
    """ etcd watch callback function
    """

    def a(event):

        my_log.function('cb_func')        
        my_log.debug("received event= {}".format(event))
        print(event)
        cmd = event['cmd']
        value = event['val']
        my_log.info("cmd= {}, value= {}".format(cmd, value))
        process(params,cmd,value,my_ds)
    
    return a
                                                                                                


def corr_run(args):

    """Main entry point. Will never return.
    :param args: Input arguments from argparse.
    """
    # open logger
    my_log.function('corr_run')

    # parse argument
    my_log.debug('config file '+args.corr_config_file)
    my_log.debug('corr node num '+str(args.corr_num))
    params = read_yaml(args.corr_config_file)
    if params is not None:
        my_log.debug('read params from config file')

    # connect to etcd
    my_ds = ds.DsaStore()

    # register watch callback on /cmd/corr/corr_num, and /cmd/corr/0
    my_ds.add_watch('/cmd/corr/'+str(args.corr_num), cb_func(params,my_ds))
    my_ds.add_watch('/cmd/corr/0', cb_func(params,my_ds))

    # infinite monitoring loop
    while True:

        key = '/mon/corr/' + str(args.corr_num)
        md = get_monitor_dict(params,args.corr_num)
        if md!=-1:
            try:
                my_ds.put_dict(key, md)
            except:
                my_log.error('COULD NOT CONNECT TO ETCD')
        sleep(2)

                                                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--corr_config_file', type=str, default='corrConfig.yaml', help='correlator config')
    parser.add_argument('-cn', '--corr_num', type=int, default='1', help='corr node number')
    the_args = parser.parse_args()
    corr_run(the_args)

    
    
