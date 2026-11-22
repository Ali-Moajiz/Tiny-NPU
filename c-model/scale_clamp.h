#ifndef SCALE_CLAMP_H
#define SCALE_CLAMP_H
#include "types.h"

/* ---- One processing element ---- */

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

/* ---- Array/container ---- */

typedef struct {
    ResultSignal result;
    bool collecting;
    uint32_t pc;
    unsigned neurons, batch, next_column;
    Line128 config;
} ScaleClampRegisters;
typedef struct {
    struct {
        const ControlBus *control;
        const AccLaneBus *lane[ARRAY_SIZE];
    } in;
    struct { ResultSignalBus *result; } out;
} ScaleClampPorts;
typedef struct {
    ScaleClampRegisters regs;
    ScaleClampPorts ports;
    ScalePe lane[ARRAY_SIZE];
    ByteLaneBus scaled[ARRAY_SIZE];
    struct { ByteLane lane[ARRAY_SIZE]; bool capture; } comb;
} ScaleClamp;
extern const BlockOps scale_clamp_ops;

#endif
