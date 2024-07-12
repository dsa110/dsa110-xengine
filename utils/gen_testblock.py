import numpy as np, struct
import matplotlib.pyplot as plt
import os

''' The aim here is to make data blocks to test the bfCorr code. 

Structure of a packet is 3 ant, 384 chan, 2 time, 2 pol, r/i
4608 bytes long

Structure of a block is [2048 packets, 32 channel groups, ...]


We want the real and imagniary parts to be random integers over 
the range of [-8, 7]
=======
'''

# defaults
outfile = 'block.out'
if os.path.exists(outfile):
    os.remove(outfile)
    

num_packets = 4
n_antennae = 3
n_chans = 384
n_changs = 32

# make values in the range vals = [-8, 7]
# [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, NANTS/2, 8chan, 2 times, 2 pol, 4-bit complex]


for ipacket in np.arange(num_packets):

    print(ipacket)
    for ichang in np.arange(n_changs):

        packet = np.zeros(num_packets*n_changs, dtype='uint8')
        for i in np.arange(n_antennae):
            for j in np.arange(n_chans):
                for k in np.arange(num_packets):

                    # we now make a randon integer iunt8 format
                    idx = ichang + n_changs*ipacket
                    packet[idx] = np.random.randint(0, 256)
                    
        out_str = packet.tobytes()        
        newFile = open(outfile, "ab")
        newFile.write(out_str)
        newFile.close()
