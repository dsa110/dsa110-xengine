import numpy as np, struct
import matplotlib.pyplot as plt


''' The aim here is to make data blocks to test the bfCorr code. 

Structure of a packet is 3 ant, 384 chan, 2 time, 2 pol, r/i
4608 bytes long

Structure of a block is [2048 packets, 32 channel groups, ...]

'''

# defaults
outfile = 'block.out'
n_packet = 4608 # 4608 for single packet
npackets = 4
nchangs = 32

# make a block where every s
chans = np.arange(384)#np.asarray([10,100,190]
v1 = 1
v2 = 2
v3 = 3
v4 = 4
v5 = 5
v6 = 6

vals = [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
# [NCHAN_PER_PACKET/8, NPACKETS_PER_BLOCK/4, 4tim, NANTS/2, 8chan, 2 times, 2 pol, 4-bit complex]

for ipacket in np.arange(npackets):

    print(ipacket)
    ant_number = 0
    for ichang in np.arange(nchangs):

        real_part = np.zeros(n_packet,dtype='int8')
        imag_part = np.zeros(n_packet,dtype='int8')

        for i in np.arange(3):
            for j in np.arange(384):
                for k in np.arange(4):

                    #v1 = 32.*(j/384.+0.8)*np.random.normal()
                    #v2 = 32.*(j/384.+0.8)*np.random.normal()

                    v1 = 32.*(((j % 9)-5)+(i+ipacket-3))
#                    if i==0:
#                        if k==0:
#                            if ipacket==0:
#                                print(j,v1/32.)
                    v2 = 0.
                    ii = i*1536+j*4+k

                    real_part[ii] = v1
                    imag_part[ii] = v2

        # make 4-bit versions
        real_part = np.cast['uint8'](real_part)
        imag_part = np.cast['uint8'](imag_part)
        for i in range(n_packet):
            real_part[i]  = real_part[i] >> 4
            imag_part[i]  = (imag_part[i] >> 4) << 4

        # finish packet
        packet = np.zeros(n_packet,dtype='uint8')
        for i in range(n_packet):
            packet[i] = real_part[i] | imag_part[i]
                
        out_str = packet.tobytes()

        newFile = open(outfile, "ab")
        newFile.write(out_str)
        newFile.close()


