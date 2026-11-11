#ifndef DATAPATH_H
#define DATAPATH_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define ARRAY_SIZE 4u
#define OPERAND_SLOTS 2u
#define SRAM_LINES 1024u
#define SRAM_BYTES 0x4000u
#define ENTRY_PC 0x3900u
#ifndef POST_LATENCY
#define POST_LATENCY 1
#endif
#if POST_LATENCY != 0 && POST_LATENCY != 1
#error POST_LATENCY must be 0 or 1
#endif
/* lane[0] is the least significant word, independent of host endianness. */
typedef struct { uint32_t lane[4]; } Line128;
typedef struct { int32_t value[4][4]; } TileSums;
typedef enum { OPERAND_X, OPERAND_W, OPERAND_B, OPERAND_CFG } OperandField;
typedef enum { WRITE_STREAM, WRITE_RESULT } WriteSource;
/* Control wires only. Addresses are BYTE addresses; no data passes through here. */
typedef struct {
    bool rd_en, read_consume, wr_en;
    uint32_t rd_addr, wr_addr;
    uint16_t byte_enable;
    WriteSource write_source;
    bool latch_en; unsigned latch_slot, latch_field;
    bool compute_en, accumulate, post_en, forward_en;
    unsigned slot, k, k_count, neurons, batch, forward_slot;
    bool input_pop, output_push;
    uint32_t pc;
} Control;
static inline int8_t unpack_i8(uint32_t word, unsigned byte) {
    int v=(int)((word>>(8*byte))&255u);
    return (int8_t)(v<128?v:v-256);
}
static inline int32_t unpack_i32(uint32_t word) {
    return word<=INT32_MAX?(int32_t)word:-1-(int32_t)(UINT32_MAX-word);
}
static inline int8_t line_byte(Line128 line, unsigned byte) {
    return unpack_i8(line.lane[byte/4],byte%4);
}
static inline void line_set_byte(Line128 *line, unsigned byte, uint8_t value) {
    unsigned shift=8*(byte%4);
    line->lane[byte/4]=(line->lane[byte/4]&~(UINT32_C(255)<<shift))|((uint32_t)value<<shift);
}
#endif
