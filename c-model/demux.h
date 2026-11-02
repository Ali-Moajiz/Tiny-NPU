#ifndef DEMUX_H
#define DEMUX_H
#include "datapath.h"
/* One-cycle response steering. Both TYPE and BANK travel with RD_EN. */
typedef struct {
    Bus out[4];
    struct { const Bus *control,*sram_q; Bus *outputs[4]; } ports;
    TypeSel type_q;
    unsigned bank_q;
    bool valid_q;
    unsigned routed[4];
    bool error;
} Demux;
bool demux_init(Demux *,BusNetwork *,const Bus *,const Bus *);
void demuxBusses(void *,uint64_t);
#endif
