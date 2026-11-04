#ifndef MAC_H
#define MAC_H
#include "signals.h"
#include "framework.h"

typedef struct
{
    struct
    {
        const ControlBus *control;
        const TileSignalBus *weights, *inputs, *biases;
    } in;
    struct
    {
        AccSignalBus *sums;
    } out;
} MacPorts;

typedef struct
{
    TileSums sums;
    bool final_valid;
    unsigned updates, compute_cycles, peak_active, next_wave;
} MacRegs;
typedef struct
{
    MacPorts ports;
    MacRegs regs;
} MacArray;

extern const BlockOps mac_ops;
#endif
