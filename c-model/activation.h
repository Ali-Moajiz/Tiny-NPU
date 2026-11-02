#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "datapath.h"
typedef struct
{
    Bus output_bus;
    struct
    {
        const Bus *control, *row_sums, *config;
        Bus *activated;
    } ports;
    bool error;
} Activation;
/* config = registered banked CFG; activation output is a registered tile. */
bool activation_init(Activation *, BusNetwork *, const Bus *control, const Bus *row_sums, const Bus *config);
void activationBusses(void *, uint64_t);
#endif
