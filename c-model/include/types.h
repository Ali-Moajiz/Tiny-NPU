#ifndef TYPES_H
#define TYPES_H
#include "bus.h"
#define ARRAY_SIZE 4u
#define OPERAND_SLOTS 2u
#define SRAM_BYTES 0x100000u
#define SRAM_LINES (SRAM_BYTES / 16u)
#define WEIGHT_BASE 0x00000u
#define WEIGHT_END 0xC0000u
#define ACTIVATION_BASE 0xC0000u
#define ACTIVATION_END 0xE0000u
#define BIAS_BASE 0xE0000u
#define BIAS_END 0xF0000u
#define CONFIG_BASE 0xF0000u
#define CONFIG_END 0xF1000u
#define ENTRY_PC 0xF1000u
/* One registered activation stage and one registered scaling stage. */
#define POST_PIPELINE_STAGES 2u
/* lane[0] is the least significant word, independent of host endianness. */
typedef struct
{
    uint32_t lane[4];
} Line128;
typedef struct
{
    int32_t value[4][4];
} TileSums;
typedef enum
{
    OPERAND_X,
    OPERAND_W,
    OPERAND_B,
    OPERAND_CFG
} OperandField;
typedef enum
{
    WRITE_STREAM,
    WRITE_RESULT
} WriteSource;
/* Control wires only. Addresses are BYTE addresses; no data passes through here. */
typedef struct
{
    bool rd_en, read_consume, wr_en;
    uint32_t rd_addr, wr_addr;
    uint16_t byte_enable;
    WriteSource write_source;
    bool latch_en;
    unsigned latch_slot, latch_field;
    bool compute_en, accumulate, post_en, forward_en;
    unsigned slot, k, k_count, neurons, batch, forward_slot, post_col;
    bool input_pop, output_push;
    uint32_t pc;
} Control;
static inline int8_t unpack_i8(uint32_t word, unsigned byte)
{
    int v = (int)((word >> (8 * byte)) & 255u);
    return (int8_t)(v < 128 ? v : v - 256);
}
static inline int32_t unpack_i32(uint32_t word)
{
    return word <= INT32_MAX ? (int32_t)word : -1 - (int32_t)(UINT32_MAX - word);
}
static inline int8_t line_byte(Line128 line, unsigned byte)
{
    return unpack_i8(line.lane[byte / 4], byte % 4);
}
static inline void line_set_byte(Line128 *line, unsigned byte, uint8_t value)
{
    unsigned shift = 8 * (byte % 4);
    line->lane[byte / 4] = (line->lane[byte / 4] & ~(UINT32_C(255) << shift)) | ((uint32_t)value << shift);
}
/* ---- Aligned physical address check ---- */

static inline bool isa_address(uint32_t address, uint32_t start, uint32_t end)
{
    return !(address & 15u) && end <= SRAM_BYTES && end >= 16u &&
           address >= start && address <= end - 16u;
}

/* ---- System bus payloads ---- */

typedef struct
{
    Line128 tile;
    bool valid;
} TileSignal;
typedef struct
{
    Line128 tile;
    bool valid;
    unsigned neurons, batch;
} ResultSignal;
typedef struct
{
    bool input_valid, output_ready;
} StreamStatus;
BUS_SIGNAL(Control)
BUS_SIGNAL(TileSignal)
BUS_SIGNAL(ResultSignal)
BUS_SIGNAL(StreamStatus)

/* ---- PE bus payloads ---- */

/* Broadcast wires: all cells in one row/column share the same bus object. */
typedef struct { int8_t value; } Operand8;
typedef struct { int32_t value; } Bias32;
typedef struct { int32_t value; bool valid; } MacResult;

/* Control travels with each registered post-processing value. These tags model
 * sideband wires, not an external bus protocol. Column=n, lane=vector. */
typedef struct
{
    Line128 config;
    uint32_t pc;
    unsigned column, neurons, batch;
} PostTag;
typedef struct { int32_t value; bool valid; PostTag tag; } AccLane;
typedef struct { int8_t value; bool valid; PostTag tag; } ByteLane;
BUS_SIGNAL(Operand8)
BUS_SIGNAL(Bias32)
BUS_SIGNAL(MacResult)
BUS_SIGNAL(AccLane)
BUS_SIGNAL(ByteLane)

/* ---- Integer rounding primitive ---- */

/* Datapath operation: signed magnitude, quotient/remainder, tie-to-even mux.
 * Callers validate shift <= 62. No floating point or signed right shift. */
static inline int64_t pe_round_shift(int64_t value, unsigned shift)
{
    if (!shift) return value;
    bool negative = value < 0;
    uint64_t magnitude = negative ? (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;
    uint64_t divisor = UINT64_C(1) << shift;
    uint64_t quotient = magnitude >> shift;
    uint64_t remainder = magnitude & (divisor - 1);
    if (remainder > divisor / 2 || (remainder == divisor / 2 && (quotient & 1)))
        quotient++;
    return negative ? -(int64_t)quotient : (int64_t)quotient;
}

#endif
