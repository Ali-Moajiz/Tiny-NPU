#ifndef PE_SIGNALS_H
#define PE_SIGNALS_H
#include "signals.h"

/* Broadcast wires: all cells in one row/column share the same bus object. */
typedef struct { int8_t value; } Operand8;
typedef struct { int32_t value; } Bias32;
typedef struct { int32_t value; bool valid; } MacResult;

/* Control travels with each registered post-processing value. These tags model
 * sideband wires, not an external bus protocol. Column=n, lane=vector. */
typedef struct
{
    Line128 config;
    uint32_t pc;
    unsigned column, neurons, batch;
} PostTag;
typedef struct { int32_t value; bool valid; PostTag tag; } AccLane;
typedef struct { int8_t value; bool valid; PostTag tag; } ByteLane;
BUS_SIGNAL(Operand8)
BUS_SIGNAL(Bias32)
BUS_SIGNAL(MacResult)
BUS_SIGNAL(AccLane)
BUS_SIGNAL(ByteLane)
#endif
