#ifndef DATAPATH_H
#define DATAPATH_H
#include "bus.h"
#define ARRAY_SIZE 4
#define NETWORK_LAYERS 2
#define COMPUTE_WAVES 4u /* column wave C1..C4; all four rows run in parallel */
/* E1..E4 read W,I,B,CFG | E5..E8 compute | E9 POST | E10 STORE */
#define LAYER_TOTAL_EDGES 10u
#define NPU_INFERENCE_CYCLES (NETWORK_LAYERS * LAYER_TOTAL_EDGES)

/* Explicit logical 128-bit value: lane[0] is bits31:0, independent of host endian. */
typedef struct
{
    uint32_t lane[4];
} Line128;

/* 2-bit Type Select driven by the controller (truth table on diagram page 01). */
typedef enum
{
    TYPE_WEIGHT = 0, /* 00 */
    TYPE_INPUT = 1,  /* 01 */
    TYPE_BIAS = 2,   /* 10 */
    TYPE_CONFIG = 3  /* 11 */
} TypeSel;

/* All bases are 10-bit SRAM LINE addresses (0x000..0x3FF), one 128-bit line each. */
typedef struct
{
    uint32_t weight_base, bias_base, input_base, output_base, config_base;
} LayerConfig;

typedef enum
{
    EDGE_LOAD,
    EDGE_COMPUTE,
    EDGE_POST,
    EDGE_STORE
} EdgeKind;

/* Dedicated control wires; SRAM data is on a separate registered 128-bit link.
 * There is no capture tag any more: the demux's own type register delays
 * read_type by the SRAM read latency and steers Q to the right register. */
typedef struct
{
    EdgeKind kind;
    bool read_enable;
    bool tile_start;  /* clears register-valid flags at the start of a tile */
    bool bias_enable; /* column-1 psum_in mux: bias[r] (1) or 0 (0) */
    TypeSel read_type;
    uint32_t address; /* line address [9:0] for SRAM read or write */
    uint16_t byte_enable;
    unsigned layer, wave; /* wave = 1..4 during EDGE_COMPUTE */
} Control;

/* Output of a 128-bit operand register: value plus a model-only "loaded" flag. */
typedef struct
{
    Line128 line;
    uint8_t loaded;
} RegValue;

typedef struct
{
    int32_t row[ARRAY_SIZE];
} RowSums;
typedef struct
{
    int8_t row[ARRAY_SIZE];
} Quantized;

static inline int8_t unpack_i8(uint32_t word, unsigned byte)
{
    int value = (int)((word >> (8 * byte)) & 255u);
    return (int8_t)(value < 128 ? value : value - 256);
}
static inline int32_t unpack_i32(uint32_t word)
{
    return word <= INT32_MAX ? (int32_t)word : -1 - (int32_t)(UINT32_MAX - word);
}
static inline uint32_t pack_i8(const int8_t data[ARRAY_SIZE])
{
    uint32_t word = 0;
    for (unsigned i = 0; i < ARRAY_SIZE; i++)
        word |= (uint32_t)(uint8_t)data[i] << (8 * i);
    return word;
}
/* The 4x4 array needs I[r][c] = x[c] for every row r, so a 4-element vector
 * is stored in all four 32-bit lanes of its 128-bit line (fixed fan-out). */
static inline Line128 line_broadcast(uint32_t word)
{
    return (Line128){{word, word, word, word}};
}
#endif
