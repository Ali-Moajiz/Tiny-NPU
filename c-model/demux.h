#ifndef DEMUX_H
#define DEMUX_H
#include "datapath.h"
/* Data steering demux 1:4 with its type register (diagram page 01).
 * The type register holds the Type Select issued with a read for one clock,
 * i.e. z^-L with L = SRAM read latency = 1. When Q arrives, the delayed type
 * selects which of the four outputs carries it. Unselected outputs stay
 * invalid (no load enable), so only the selected register latches. */
typedef struct
{
    Bus out[4]; /* wires, indexed by TypeSel */
    struct
    {
        const Bus *control, *sram_q;
        Bus *outputs[4];
    } ports;
    TypeSel type_q;
    bool valid_q;
    unsigned routed[4];
    bool error;
} Demux;

bool demux_init(Demux *, BusNetwork *, const Bus *control, const Bus *sram_q);
void demuxBusses(void *, uint64_t);
#endif
