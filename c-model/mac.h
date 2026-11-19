#ifndef MAC_H
#define MAC_H
#include "mac_pe.h"

typedef struct { Line128 operand[4]; bool loaded[4]; } OperandSlot;
typedef struct {
    OperandSlot slot[OPERAND_SLOTS];
    bool valid, complete;
    unsigned neurons, batch, next_k, k_count, compute_slot, post_next;
    uint64_t useful_macs, compute_cycles;
} MacRegisters;
typedef struct {
    struct {
        const ControlBus *control;
        const TileSignalBus *response;
        const ResultSignalBus *forward;
    } in;
    struct { AccLaneBus *lane[ARRAY_SIZE]; } out;
} MacPorts;
typedef struct {
    MacRegisters regs;
    MacPorts ports;
    MacPe cell[ARRAY_SIZE][ARRAY_SIZE]; /* [vector][neuron] */
    struct {
        Operand8Bus x[ARRAY_SIZE], w[ARRAY_SIZE];
        Bias32Bus bias[ARRAY_SIZE];
        MacResultBus sum[ARRAY_SIZE][ARRAY_SIZE];
    } wires;
} Mac;
extern const BlockOps mac_ops;
#endif
