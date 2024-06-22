import socket, numpy as np

# ip as string, port as int, buf as int
def capture(n=100,ip=None,port=None,buf=4616):

    if ip is None:
        print('No IP')
        return()

    if port is None:
        print('No port')
        return()

    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((ip,port))

    captured=0
    packs = []
    while captured<n:

        data, addr = sock.recvfrom(buf)
        packs.append(data)
        captured += 1


    return(packs)



        
        
