#!/bin/bash
#

while [ 1 ]; do

    now=`date -I`
    for fl in `ls /dataz/dsa110/operations/correlator/${now}*_spl.hdf5`; do
	rsync -avv --remove-source-files ${fl} ubuntu@dsacamera.ovro.pvt:/data/incoming
    done
    for fl in `ls /dataz/dsa110/operations/correlator/${now}*_sb??.hdf5`; do
	rsync -avv ${fl} ubuntu@dsacamera.ovro.pvt:/data/incoming
    done

    sleep 180

done


