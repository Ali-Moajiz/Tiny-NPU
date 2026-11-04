#ifndef SCALE_CLAMP_H
#define SCALE_CLAMP_H
#include "signals.h"
#include "framework.h"

typedef struct {
    struct { const ControlBus *control; const AccSignalBus *sums; const TileSignalBus *config; } in;
    struct { TileSignalBus *result; } out;
} PostPorts;
typedef struct { Line128 tile; bool valid; } PostRegs;
typedef struct {
    PostPorts ports;
#if POST_LATENCY == 1
    PostRegs regs;
#endif
} PostProcess;

int8_t scale_clamp_value(int32_t, int32_t, unsigned);
extern const BlockOps post_ops;
#endif
