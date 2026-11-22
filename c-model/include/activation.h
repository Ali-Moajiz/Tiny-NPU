#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "types.h"

/* ---- One processing element ---- */

typedef enum {
    ACT_BYPASS=0, ACT_RELU=1, ACT_LEAKY_RELU=2, ACT_CLIPPED_RELU=3
} ActivationMode;
/* CFG lane[2]=mode; lane[3]=0 for bypass/ReLU, right shift 0..31 for
 * leaky ReLU (slope=2^-shift), positive INT32 ceiling for clipped ReLU.
 * Values remain in accumulator units; clipping thresholds are NOT floats. */
bool activation_config_valid(Line128 config);
bool activation_value(int32_t input, unsigned mode, uint32_t parameter, int32_t *output);
const char *activation_name(unsigned mode);
typedef struct { AccLane output; } ActivationPeRegisters;
typedef struct {
    struct { const AccLaneBus *value; } in;
    struct { AccLaneBus *value; } out;
} ActivationPePorts;
typedef struct {
    ActivationPeRegisters regs;
    ActivationPePorts ports;
    unsigned lane;
    struct { AccLane result; } comb;
} ActivationPe;
extern const BlockOps activation_pe_ops;

/* ---- Array/container ---- */

typedef struct {
    struct { const AccLaneBus *lane[ARRAY_SIZE]; } in;
    struct { AccLaneBus *lane[ARRAY_SIZE]; } out;
} ActivationPorts;
/* Bank container; each lane owns its actual register and callbacks. */
typedef struct { ActivationPorts ports; ActivationPe lane[ARRAY_SIZE]; } Activation;
void activation_init(Activation *);

#endif
