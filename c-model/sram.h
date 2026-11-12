#ifndef SRAM_H
#define SRAM_H
#include "framework.h"
#include "signals.h"

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

/* Host initialization only: exactly SRAM_LINES MSB-first 128-bit hex words. */
bool sram_load_hex(Sram *, const char *path, char *error, size_t error_size);
extern const BlockOps sram_ops;
#endif
