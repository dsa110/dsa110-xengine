#!/bin/bash

dir="20feb21"

while true; do

    echo "Starting..."
    dsacon corr start
    sleep 20
    echo "Setting..."
    dsacon corr set
    sleep 3
    
    echo "Sleeping 3600s"
    sleep 3600
    echo "Stopping..."
    dsacon corr stop
    sleep 30

    # run copy
    ssh user@dsastorage.ovro.pvt "source ~/.bashrc; cd /mnt/data/dsa110/T1/; ./copy_T1.bash ${dir}; cd ../T2; ./copy_T2.bash ${dir}; cd ../T3; ./copy_T3.bash ${dir}"

    sleep 1
    
    # gen plots
    ssh user@dsastorage.ovro.pvt "source ~/.bashrc; cd /mnt/data/dsa110/T3/scripts; python gen_T3_inspect.py ${dir}"

    sleep 1
    
       
done
