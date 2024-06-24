import numpy as np
import sockets as s
import struct
import sys
import matplotlib.pyplot as plt

# for file writing

def write_bin(data,fl='test.dat'):

        f = open(fl,'w+b')
        for packet in data:
                d = bytearray(np.asarray(struct.unpack('>4616B',packet))[8:].astype(np.int8))
                print(len(d))
                f.write(d)

        f.close()
        

# for making histogram of input

def make_histogram(data,ant=0,pol=0):

        histo = np.zeros(16)
        rms = 0.
        
        for packet in data:
                
                d = np.asarray(struct.unpack('>4616B',packet))[8:]
                
                # order is 3 antennas x 384 channels x 2 times x 2 pols x real/imag, with every 8 flipped
                d = (d.reshape((3,384,2,2)))[ant,:,:,pol].ravel()
                
                d_r = ((d & 15) << 4)
                d_i = d & 240
                d_r = d_r.astype(np.int8)/16
                d_i = d_i.astype(np.int8)/16        

                rms += 0.5*(np.std(d_r)**2.+np.std(d_i)**2.)
                
                for i in range(384*2):

                        histo[int(d_r[i])+8] += 1.
                        histo[int(d_i[i])+8] += 1.
                                                                        
        return histo/np.max(histo),np.sqrt(rms)
        
# for making spectrum from data
def decode_data(data,ant=0,pol=0):

    spec = np.zeros(384*2)
    
    for packet in data:

        d = np.asarray(struct.unpack('>4616B',packet))[8:]

        # order is 3 antennas x 384 channels x 2 times x 2 pols x real/imag, with every 8 flipped
        d = (d.reshape((3,384,2,2)))[ant,:,:,pol].ravel()

        d_r = ((d & 15) << 4)
        d_i = d & 240
        d_r = d_r.astype(np.int8)/16
        d_i = d_i.astype(np.int8)/16     
        
        spec += d_r**2.+d_i**2.

    spec = spec.reshape((384,2)).mean(axis=1)
    return(spec)

# for decoding packets
def decode_header(data):

    min_s = 10000
    max_s = 0
        
    for packet in data:

        d = np.asarray(struct.unpack('>4616B',packet))

        # packet id
        p = 0
        p = p | ((d[4] & 224) >> 5)
        p = p | (d[3] << 3)
        p = p | (d[2] << 11)
        p = p | (d[1] << 19)
        p = p | (d[0] << 27)
        
        # spectrum id
        sp = 0
        sp = sp | ((d[4] & 31) << 8)
        sp = sp | d[5]

        if (sp<min_s):
                min_s = sp
        if (sp>max_s):
                max_s = sp
    
        print(p,sp)

    print(min_s,max_s)

# MAIN

n = 10000
ip = '10.41.0.62'
port=4011
data = s.capture(ip=ip,port=port,n=n)
ant=0
pol=0

#decode_header(data)

histo,rms = make_histogram(data,ant=ant,pol=pol)
print()
print('RMS:',rms/np.sqrt(1.*n))
for i in np.arange(16):
    print(histo[i],'  ',)

sys.exit()
    
spec = decode_data(data,ant=ant,pol=pol)
spec = np.sqrt(spec/n/2.)
print()
print('Have spectral points',len(spec))
print()
#for i in np.arange(len(spec)):
#    print(spec[i],'  ',)

plt.plot(spec)
plt.show()





    


    

    
