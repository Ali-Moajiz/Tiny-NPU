#ifndef ISA_H
#define ISA_H
#include "datapath.h"
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
bool isa_address(uint32_t address, uint32_t start, uint32_t end);
static inline unsigned instruction_n(Instruction i) { return i.dims & 15; }
static inline unsigned instruction_v(Instruction i) { return (i.dims >> 4) & 15; }
static inline unsigned instruction_k(Instruction i) { return (i.dims >> 8) & 15; }
#endif
