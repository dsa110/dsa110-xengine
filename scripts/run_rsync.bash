#!/bin/bash
#

for i in `seq 1 5`; do

    rsync --partial --timeout=20 -rlpgoDvz ${1} ${2}
    if [ "$?" -eq "0" ]; then
	break
    fi
    
done

