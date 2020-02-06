#ifndef __DSAX_DEF_H
#define __DSAX_DEF_H

#include "dada_def.h"

// default dada block keys
#define TEST_BLOCK_KEY 0x0000aada // for capture program.
// 128*3*384*32*2*8=75497472 for 1 CHANG 1 SNAP
#define CAPTURE_BLOCK_KEY 0x0000dada // for capture program.
// 128*3*384*32*2*8=75497472 for 1 CHANG 1 SNAP
#define REORDER_BLOCK_KEY 0x0000eada // for reorder program.
// 128*32*1536*32*2*2=805306368 for 1 CHANG
#define XGPU_BLOCK_KEY 0x0000fada // for xgpu program. 
// 136*1536*2*8=3342336 for 1 CHANG 

// default number of XGPU ints
#define NCORRINTS 128
#define NNATINTS 32 // native number of integrations
#define NREORDERS 8 // number of ints per reorder

// size of xgpu output
// TODO
#define XGPU_SIZE 835584 // size of single output vector (post-GPU)
#define XGPU_IN_INC 1 // size of input increment
#define NBASE 136 // nant*(nant+1)/2
#define NPOL 2
#define NCHAN 1536 // regardless of NCHANG
#define NANT 16 // number of antennas in XGPU

// default port for packet capture
#define CAPTURE_PORT 4011

// default UDP packet dims
#define UDP_HEADER   8              // size of header/sequence number
#define UDP_DATA     4608           // obs bytes per packet
#define UDP_PAYLOAD  4616           // header + datasize

// number of channel groups to expect
#define NCHANG 1

// number of SNAPs to expect
#define NSNAPS 1

/* expect consecutive channel groups */
#define CHOFF 4608 // offset in channels of first group

// default control ports
#define CAPTURE_CONTROL_PORT 11223
#define REORDER_CONTROL_PORT 11224
#define XGPU_CONTROL_PORT 11225
#define WRITEVIS_CONTROL_PORT 11226


#endif 

