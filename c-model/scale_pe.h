#ifndef SCALE_PE_H
#define SCALE_PE_H
#include "framework.h"
#include "pe_signals.h"
int8_t scale_clamp_value(int32_t value, int32_t multiplier, unsigned shift);
typedef struct { ByteLane output; } ScalePeRegisters;
typedef struct {
    struct { const AccLaneBus *value; } in;
    struct { ByteLaneBus *value; } out;
} ScalePePorts;
typedef struct {
    ScalePeRegisters regs;
    ScalePePorts ports;
    unsigned lane;
    struct {
        int64_t product, rounded; /* 32x32 multiplier, rounded barrel shift. */
        ByteLane result;         /* Signed INT8 saturation mux output. */
    } comb;
} ScalePe;
extern const BlockOps scale_pe_ops;
#endif
