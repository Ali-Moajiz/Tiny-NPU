#ifndef REG128_H
#define REG128_H
#include "datapath.h"
/* Two physical 128-bit banks per object. Registered Q is published every edge;
 * a consumer never observes a value captured at that SAME edge.
 * Input bank D mux: tagged SRAM response OR registered requantized hidden tile.
 * Parameter bank D: tagged SRAM response. Unselected banks retain their bits. */
typedef struct {
    const char *name;
    TypeSel type;
    Bus out_bus;
    struct { const Bus *control,*in,*forwarded; Bus *out; } ports;
    RegBanks value;
    unsigned captures,forwards;
    bool error;
} Reg128;
bool reg128_init(Reg128 *,BusNetwork *,const char *,TypeSel,const Bus *,const Bus *,const Bus *);
void reg128Busses(void *,uint64_t);
#endif
