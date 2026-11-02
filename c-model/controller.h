#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "datapath.h"
typedef enum { CTRL_COLD,CTRL_READY,CTRL_COMPUTE,CTRL_ACTIVATE,
               CTRL_SCALE,CTRL_HANDOFF,CTRL_STORE,CTRL_RELOAD_INPUT } ControllerState;
typedef struct {
    Bus control_bus;
    struct {
        Bus *control;
        const Bus *fifo_valid,*weights,*inputs,*biases,*config;
    } ports;
    LayerConfig layers[NETWORK_LAYERS];
    NpuOptions options;
    ControllerState state;
    unsigned layer,cold_read,wave,issued[NETWORK_LAYERS];
    unsigned wait_cycles,overlap_reads;
    bool done,error;
} Controller;
bool controller_init(Controller *,BusNetwork *,const LayerConfig[NETWORK_LAYERS],
                     NpuOptions,const Bus *,const Bus *,const Bus *,const Bus *,const Bus *);
void controllerBusses(void *,uint64_t);
#endif
