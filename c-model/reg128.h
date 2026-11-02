#ifndef REG128_H
#define REG128_H
#include "datapath.h"
/* One 128-bit operand register (Weight, Input, Bias or Config).
 * D input  : the demux output for its type (valid only when it is selected = load enable).
 * Q output : registered RegValue, republished every clock, so consumers always see
 *            the OLD value on the clock in which a new one is captured. */
typedef struct
{
    const char *name;
    TypeSel type;
    Bus out_bus;

    struct
    {
        const Bus *control, *in;
        Bus *out;
    } ports;
    
    RegValue value;
    unsigned captures;
    bool error;
} Reg128;
bool reg128_init(Reg128 *, BusNetwork *, const char *name, TypeSel, const Bus *control, const Bus *demux_out);
void reg128Busses(void *, uint64_t);
#endif
