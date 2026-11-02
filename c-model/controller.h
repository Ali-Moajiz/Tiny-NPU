#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "datapath.h"

typedef struct
{
    Bus control_bus;
    struct
    {
        Bus *control;
        const Bus *fifo_valid;
    } ports;
    LayerConfig layers[NETWORK_LAYERS];
    unsigned layer, edge;
    bool done, error;
} Controller;

bool controller_init(Controller *, BusNetwork *, const LayerConfig[NETWORK_LAYERS], const Bus *fifo_valid);
void controllerBusses(void *, uint64_t);
#endif
