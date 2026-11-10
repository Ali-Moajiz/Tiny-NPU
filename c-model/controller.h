#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "signals.h"
#include "framework.h"

#define SCHEDULE_CAPACITY 32u
typedef struct
{
    bool valid;
    TypeSel type;
    unsigned bank;
} ReadTag;
typedef struct
{
    unsigned pc, overlap_reads;
    ReadTag tag;
    bool done;
} ControllerRegs;
typedef struct
{
    struct
    {
        ControlBus *control;
    } out;
} ControllerPorts;
typedef struct
{
    LayerConfig layers[NETWORK_LAYERS];
    NpuOptions options;
    Control words[SCHEDULE_CAPACITY];
    unsigned length;
    ControllerRegs regs;
    ControllerPorts ports;
} Controller;

bool controller_init(Controller *, const LayerConfig *, NpuOptions);
bool controller_validate(const Controller *);
extern const BlockOps controller_ops;
#endif
