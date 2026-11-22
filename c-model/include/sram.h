#ifndef SRAM_H
#define SRAM_H
#include "types.h"
typedef struct
{
    Line128 memory[SRAM_LINES];
    TileSignal response;
    uint64_t reads, writes;
} SramRegisters;

typedef struct
{
    struct
    {
        const ControlBus *control;
        const TileSignalBus *input;
        const ResultSignalBus *result;
    } in;
    struct
    {
        TileSignalBus *response;
    } out;
} SramPorts;

typedef struct
{
    SramRegisters regs;
    SramPorts ports;
} Sram;

extern const BlockOps sram_ops;
#endif
