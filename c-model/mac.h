#ifndef MAC_H
#define MAC_H
#include "datapath.h"
/* Each cell owns only a registered INT32 partial sum. W[r][c] and I[r][c] are
 * NOT copied into the cell: they are fixed byte lanes (4r+c) of the Weight and
 * Input registers. bias[r] reaches column 1 through a 2:1 mux (bias or 0).
 * The horizontal psum link is registered, not a combinational chain. */
typedef struct { int32_t psum;bool valid; } Mac;
typedef struct {
    Mac cell[ARRAY_SIZE][ARRAY_SIZE];
    RowSums sums;
    Bus sums_bus;
    struct { const Bus *control,*weights,*inputs,*biases;Bus *row_sums; } ports;
    bool error;
    unsigned updates;
} MacArray;
bool mac_array_init(MacArray *,BusNetwork *,const Bus *control,const Bus *weight_reg,const Bus *input_reg,const Bus *bias_reg);
void macArrayBusses(void *,uint64_t);
#endif
