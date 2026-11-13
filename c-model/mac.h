#ifndef MAC_H
#define MAC_H
#include "framework.h"
#include "signals.h"

typedef struct
{
    Line128 operand[4];
    bool loaded[4];
} OperandSlot;

typedef struct
{
    OperandSlot slot[OPERAND_SLOTS];
    TileSums accumulators;
    bool valid, complete;
    unsigned neurons, batch, next_k;
    uint64_t useful_macs, compute_cycles;
} MacRegisters;

typedef struct
{
    struct
    {
        const ControlBus *control;
        const TileSignalBus *response;
        const ResultSignalBus *forward;
    } in;
    struct
    {
        AccSignalBus *sums;
    } out;
} MacPorts;

typedef struct
{
    MacRegisters regs;
    MacPorts ports;
    TileSums products_added; /* Combinational adder results, not a q/d bank. */
} Mac;
extern const BlockOps mac_ops;
#endif
