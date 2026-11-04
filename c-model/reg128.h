#ifndef REG128_H
#define REG128_H
#include "signals.h"
#include "framework.h"

typedef struct
{
    struct
    {
        const ControlBus *control;
        const TileSignalBus *response, *hidden;
    } in;
    struct
    {
        TileSignalBus *selected;
    } out;
} Reg128Ports;

typedef struct
{
    RegBanks banks;
    unsigned captures, forwards;
} Reg128Regs;
typedef struct
{
    Reg128Ports ports;
    Reg128Regs regs;
    const char *name;
    TypeSel type;
} Reg128;

extern const BlockOps reg128_ops;
#endif
