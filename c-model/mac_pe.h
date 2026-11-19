#ifndef MAC_PE_H
#define MAC_PE_H
#include "framework.h"
#include "pe_signals.h"

/* One physical multiplier/accumulator cell. Operand registers live in its
 * owning MAC block; shared row/column buses feed this combinational datapath. */
typedef struct { int32_t accumulator; bool valid; } MacPeRegisters;
typedef struct
{
    struct {
        const Operand8Bus *x, *w;
        const Bias32Bus *bias;
        const ControlBus *control;
    } in;
    struct { MacResultBus *sum; } out;
} MacPePorts;
typedef struct
{
    MacPeRegisters regs;
    MacPePorts ports;
    unsigned vector, neuron; /* Static instance coordinates. */
    struct {
        bool enabled;
        int8_t x, w;
        int16_t product;     /* Signed 8x8 multiplier output. */
        int32_t addend;      /* Bias/feedback selection mux output. */
        int64_t wide_sum;    /* Guard width for INT32 overflow detection. */
    } comb;
} MacPe;
extern const BlockOps mac_pe_ops;
#endif
