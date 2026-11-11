#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "framework.h"
#include "signals.h"

typedef struct {
    struct {
        const ControlBus *control;
        const AccSignalBus *sums;
    } in;
    struct { AccSignalBus *activated; } out;
} ActivationPorts;

/* Purely combinational: no architectural registers are needed here. */
typedef struct { ActivationPorts ports; } Activation;
extern const BlockOps activation_ops;
#endif
