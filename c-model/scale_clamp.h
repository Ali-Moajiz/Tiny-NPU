#ifndef SCALE_CLAMP_H
#define SCALE_CLAMP_H
#include "scale_pe.h"
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
