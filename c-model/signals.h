#ifndef SIGNALS_H
#define SIGNALS_H
#include "datapath.h"
#include "bus.h"

/* A software payload groups related wires. Valid is a separate sideband bit. */
typedef struct
{
    Line128 tile;
    bool valid;
} TileSignal;
typedef struct
{
    TileSums tile;
    bool valid;
} AccSignal;
BUS_SIGNAL(Control)
BUS_SIGNAL(TileSignal)
BUS_SIGNAL(AccSignal)
#endif
