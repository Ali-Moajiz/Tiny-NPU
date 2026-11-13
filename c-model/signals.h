#ifndef SIGNALS_H
#define SIGNALS_H
#include "datapath.h"
#include "bus.h"
typedef struct
{
    Line128 tile;
    bool valid;
} TileSignal;
typedef struct
{
    Line128 tile;
    bool valid;
    unsigned neurons, batch;
} ResultSignal;
typedef struct
{
    TileSums tile;
    Line128 config;
    bool valid;
    unsigned neurons, batch;
} AccSignal;
typedef struct
{
    bool input_valid, output_ready;
} StreamStatus;
BUS_SIGNAL(Control)
BUS_SIGNAL(TileSignal)
BUS_SIGNAL(ResultSignal)
BUS_SIGNAL(AccSignal)
BUS_SIGNAL(StreamStatus)
#endif
