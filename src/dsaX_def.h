#ifndef __DSAX_DEF_H
#define __DSAX_DEF_H

#include "dada_def.h"

// default dada block keys
#define CAPTURE_BLOCK_KEY 0x0000dada // for capture program
#define REORDER_BLOCK_KEY 0x0000eada // for reorder program
#define XGPU_BLOCK_KEY 0x0000eada // for reorder program

// default number of XGPU ints
#define NCORRINTS 128
#define NNATINTS 32 // native number of integrations

// size of xgpu output
// TODO
#define XGPU_SIZE 3244032 // size of single output vector (post-GPU)
#define XGPU_IN_INC 1 // size of input increment
#define NBASE 528 // nant*(nant+1)/2
#define NPOL 2
#define NCHAN 384 // change depending on NCHANG
#define NANT 32 // number of antennas in XGPU

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

// default control ports
#define CAPTURE_CONTROL_PORT 11223
#define REORDER_CONTROL_PORT 11224
#define XGPU_CONTROL_PORT 11225


#endif 

