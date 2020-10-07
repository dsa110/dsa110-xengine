import numpy as np

# size is 64 antennas, then 64, NW, 2pol, ri
NW = 48
outfile = 'antennas.out'

antpos = np.arange(64,dtype=np.float32)*6.25
weights = np.ones(64*NW*2*2,dtype=np.float32)
#weights = np.zeros(64*NW*2*2,dtype=np.float32)
#weights[2*NW*2*2:3*NW*2*2] += 1.

output = np.zeros(64+64*NW*2*2,dtype=np.float32)
output[0:64] = antpos
output[64:] = weights

out_str = output.tobytes()


newFile = open(outfile, "wb")
newFile.write(out_str)
newFile.close()
