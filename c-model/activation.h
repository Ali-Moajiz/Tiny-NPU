#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "activation_pe.h"
typedef struct {
    struct { const AccLaneBus *lane[ARRAY_SIZE]; } in;
    struct { AccLaneBus *lane[ARRAY_SIZE]; } out;
} ActivationPorts;
/* Bank container; each lane owns its actual register and callbacks. */
typedef struct { ActivationPorts ports; ActivationPe lane[ARRAY_SIZE]; } Activation;
void activation_init(Activation *);
#endif
