#ifndef SCALE_CLAMP_H
#define SCALE_CLAMP_H
#include "framework.h"
#include "signals.h"

typedef struct { ResultSignal result; } ScaleClampRegisters;
typedef struct {
    struct {
        const ControlBus *control;
        const AccSignalBus *activated;
    } in;
    struct { ResultSignalBus *result; } out;
} ScaleClampPorts;

typedef struct {
    ScaleClampRegisters regs;
    ScaleClampPorts ports;
    ResultSignal evaluated; /* Combinational result before capture. */
} ScaleClamp;
int8_t scale_clamp_value(int32_t value, int32_t multiplier, unsigned shift);
extern const BlockOps scale_clamp_ops;
#endif
