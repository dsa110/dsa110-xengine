import numpy as np, struct
import matplotlib.pyplot as plt


''' The aim here is to make two types of data packets: 
 - one with a tone at a particular frequency and set of antennas
 - one with pure noise 

Structure is 3 ant, 384 chan, 2 time, 2 pol, r/i
4608 bytes long

'''


def make_spectrum(packet,ant=0,pol=0):

    spec = np.zeros(384*2)
    
    d = np.asarray(struct.unpack('>4608B',packet))

    # order is 3 antennas x 384 channels x 2 times x 2 pols x real/imag, with every 8 flipped
    d = (d.reshape((3,384,2,2)))[ant,:,:,pol].ravel()

    d_r = ((d & 15) << 4)
    d_i = d & 240
    d_r = d_r.astype(np.int8)/16
    d_i = d_i.astype(np.int8)/16     
        
    spec += d_r**2.+d_i**2.
    spec = spec.reshape((384,2)).mean(axis=1)
    return(spec)

def plot_spectrum(data,ant=0,pol=0):

    spec = make_spectrum(data,ant=ant,pol=pol)
    plt.plot(spec)
    plt.xlabel('Channel')
    plt.ylabel('Power')
    plt.show()

def make_histogram(packet):
    ''' Makes histogram of packet - tested 
    '''
    
    histo = np.zeros(16)
    rms = 0.
                
    d = np.asarray(struct.unpack('>4608B',packet))
    
    # order is 3 antennas x 384 channels x 2 times x 2 pols x real/imag, with every 8 flipped
    d = (d.reshape((3,384,2,2))).ravel()
    
    d_r = ((d & 15) << 4)
    d_i = d & 240
    d_r = d_r.astype(np.int8)/16
    d_i = d_i.astype(np.int8)/16        
    
    rms += 0.5*(np.std(d_r)**2.+np.std(d_i)**2.)

    hx = np.arange(16)-8
    
    for i in range(384*2):
        
        histo[int(d_r[i])+8] += 1.
        histo[int(d_i[i])+8] += 1.
            
    return(hx,histo/np.max(histo),np.sqrt(rms))

def histo_test(data):

    hx,histo,rms = make_histogram(data)
    print('HISTOGRAM: ')
    for i in range(16):
        print(hx[i],histo[i])
    print()
    print('RMS = ',rms)
    print()


########## MAIN ############

# defaults
outfile = 'packet.out'
n_packet = 4608 # 4608 for single packet

# decide which sort of packet to make
noise = True
tone = False
x16 = False

# if tone
if tone is True:

    # defaults:
    chans = np.arange(384)#np.asarray([10,100,190])
    #ant = 1
    amp_A = 9.0
    amp_B = 4.

    # derived quantities
    amp_A = 16.*np.sqrt(amp_A)
    amp_B = 16.*np.sqrt(amp_B)
    ph = 2.*np.pi*np.random.uniform()
    ramp_A = amp_A*np.cos(ph)
    iamp_A = amp_A*np.sin(ph)
    ph = 2.*np.pi*np.random.uniform()
    ramp_B = amp_B*np.cos(ph)
    iamp_B = amp_B*np.sin(ph)
    
    # make packet
    real_part = np.zeros(n_packet,dtype='int8')
    imag_part = np.zeros(n_packet,dtype='int8')
    for ant in [0,1,2]:
        for i in chans:

            # time 1 pol A
            j = int(1536*ant + i*4)
            real_part[j] = round(ramp_A)
            imag_part[j] = round(iamp_A)
            
            # time 1 pol B
            j = int(1536*ant + i*4 + 1)
            real_part[j] = round(ramp_B)
            imag_part[j] = round(iamp_B)
            
            # time 2 pol A
            j = int(1536*ant + i*4 + 2)
            real_part[j] = round(ramp_A)
            imag_part[j] = round(iamp_A)

            # time 2 pol B
            j = int(1536*ant + i*4 + 3)
            real_part[j] = round(ramp_B)
            imag_part[j] = round(iamp_B)

        
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

    # if x16
    if (x16):

        p2 = np.zeros(21*n_packet,dtype='uint8')
        for i in range(21):
            p2[i*n_packet:(i+1)*n_packet] = packet
    
        out_str = p2.tobytes()

    else:

        out_str = packet.tobytes()
    
# if noise
if noise is True:

    # defaults
    rms = 1.5 # 4-bit
    erms = rms*16.

    # make real and imag parts
    real_part = np.zeros(n_packet,dtype='int8')
    imag_part = np.zeros(n_packet,dtype='int8')

    for ant in [0, 1, 2]:
        for i in np.arange(384):

            # time 1 pol A
            j = int(1536*ant + i*4)
            real_part[j] = round(np.random.normal()*erms)
            imag_part[j] = round(np.random.normal()*erms)
            
            # time 1 pol B
            j = int(1536*ant + i*4 + 1)
            real_part[j] = round(np.random.normal()*erms)
            imag_part[j] = round(np.random.normal()*erms)
            
            # time 2 pol A
            j = int(1536*ant + i*4 + 2)
            real_part[j] = round(np.random.normal()*erms)
            imag_part[j] = round(np.random.normal()*erms)

            # time 2 pol B
            j = int(1536*ant + i*4 + 3)
            real_part[j] = round(np.random.normal()*erms)
            imag_part[j] = round(np.random.normal()*erms)

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


newFile = open(outfile, "wb")
newFile.write(out_str)
newFile.close()

    
#plot_spectrum(out_str,pol=1,ant=1)


    



    
        
    
    
        
    
