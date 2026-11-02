#ifndef MAC_H
#define MAC_H
#include "datapath.h"
/* OUTER-PRODUCT mapping: cell[r][v] owns one output neuron's accumulator
 * for independent input vector v. Its accumulator feeds back locally.
 * W[r,k] broadcasts along row r; X[v,k] broadcasts along vector column v.
 * This is deliberately NOT the old W[r,k] stationary/psum-right mapping. */
typedef struct { int32_t psum; bool valid; } Mac;
typedef struct {
    Mac cell[4][4];
    TileSums sums;
    Bus sums_bus;
    struct { const Bus *control,*weights,*inputs,*biases; Bus *row_sums; } ports;
    unsigned updates,compute_cycles,peak_active,next_wave;
    bool error;
} MacArray;
bool mac_array_init(MacArray *,BusNetwork *,const Bus *,const Bus *,const Bus *,const Bus *);
void macArrayBusses(void *,uint64_t);
#endif
