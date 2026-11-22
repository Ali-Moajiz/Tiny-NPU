#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "types.h"

/* ---- 128-bit ISA ---- */

typedef enum
{
    OP_STREAM_LOAD = 1,
    OP_EXEC_MAC = 2,
    OP_POST_PROC = 3,
    OP_FORWARD = 4,
    OP_STREAM_STORE = 5,
    OP_LOOP_START = 6,
    OP_LOOP_END = 7,
    OP_LOAD_OPERAND = 8,
    OP_HALT = 15
} Opcode;
typedef struct
{
    unsigned opcode;
    uint32_t src, dest, dims, flags;
} Instruction;
Instruction instruction_decode(Line128 raw);
static inline unsigned instruction_n(Instruction i) { return i.dims & 15; }
static inline unsigned instruction_v(Instruction i) { return (i.dims >> 4) & 15; }
static inline unsigned instruction_k(Instruction i) { return (i.dims >> 8) & 15; }

/* ---- Controller state and ports ---- */

typedef struct Controller Controller;
typedef enum
{
    CTRL_FETCH,
    CTRL_RESPONSE,
    CTRL_DECODE,
    CTRL_DISPATCH,
    CTRL_UPDATE
} ControllerState;
typedef enum
{
    EXEC_PROGRESS,
    EXEC_WAIT,
    EXEC_DONE,
    EXEC_FAULT
} ExecStatus;
typedef struct
{
    TileSignal memory;
    StreamStatus stream;
} ControllerInputs;
/* Descriptors must outlive the controller. execute() emits wires; only retire()
 * runs at an edge. The generic FSM never switches on an opcode. */
typedef struct
{
    unsigned opcode;
    const char *name;
    bool (*validate)(Instruction);
    ExecStatus (*execute)(const Controller *, const ControllerInputs *, Control *);
    void (*retire)(Controller *);
} InstructionHandler;
typedef struct
{
    uint32_t pc, next_pc;
    Line128 ir;
    Instruction instruction;
    ControllerState state;
    unsigned step;
    bool done, loop_active;
    uint32_t loop_remaining, loop_pc;
    uint64_t cycles, instructions, bus_stall_cycles;
} ControllerRegisters;
typedef struct
{
    struct
    {
        const TileSignalBus *memory;
        const StreamStatusBus *stream;
    } in;
    struct
    {
        ControlBus *control;
    } out;
} ControllerPorts;

struct Controller
{
    ControllerRegisters regs;
    ControllerPorts ports;
    const InstructionHandler *handlers[16];
    const InstructionHandler *active_handler;
    /* Combinational evaluation scratch, not a second register bank. */
    Instruction decoded;
    ExecStatus execution;
    unsigned expected_batch; /* optional host-side ABI guard, zero disables */
};
void controller_init(Controller *);
bool controller_register(Controller *, const InstructionHandler *);
bool controller_register_builtins(Controller *);
extern const BlockOps controller_ops;

#endif
