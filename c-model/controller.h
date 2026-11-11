#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "framework.h"
#include "isa.h"
#include "signals.h"
typedef struct Controller Controller;
typedef enum { CTRL_FETCH, CTRL_RESPONSE, CTRL_DECODE, CTRL_DISPATCH, CTRL_UPDATE } ControllerState;
typedef enum { EXEC_PROGRESS, EXEC_WAIT, EXEC_DONE, EXEC_FAULT } ExecStatus;
typedef struct { TileSignal memory; StreamStatus stream; } ControllerInputs;
/* Descriptors must outlive the controller. execute() emits wires; only retire()
 * runs at an edge. The generic FSM never switches on an opcode. */
typedef struct {
    unsigned opcode;
    const char *name;
    bool (*validate)(Instruction);
    ExecStatus (*execute)(const Controller *, const ControllerInputs *, Control *);
    void (*retire)(Controller *);
} InstructionHandler;
typedef struct {
    uint32_t pc, next_pc; Line128 ir; Instruction instruction;
    ControllerState state; unsigned step;
    bool done, loop_active;
    uint32_t loop_remaining, loop_pc;
    uint64_t cycles, instructions, bus_stall_cycles;
} ControllerRegisters;
typedef struct {
    struct { const TileSignalBus *memory; const StreamStatusBus *stream; } in;
    struct { ControlBus *control; } out;
} ControllerPorts;
struct Controller {
    ControllerRegisters regs;
    ControllerPorts ports;
    const InstructionHandler *handlers[16];
    const InstructionHandler *active_handler;
    /* Combinational evaluation scratch, not a second register bank. */
    Instruction decoded; ExecStatus execution;
    unsigned expected_batch; /* optional host-side ABI guard, zero disables */
};
void controller_init(Controller *);
bool controller_register(Controller *, const InstructionHandler *);
bool controller_register_builtins(Controller *);
extern const BlockOps controller_ops;
#endif
