#ifndef DATAPATH_H
#define DATAPATH_H
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#define ARRAY_SIZE 4u
#define NETWORK_LAYERS 2u
#define PARAM_BANKS 2u
#define COMPUTE_WAVES 4u
/* Safety bound, not a timing claim. Completion comes from the controller schedule. */
#define NPU_INFERENCE_CYCLES 128u
#ifndef NPU_BATCH
#define NPU_BATCH 1u
#endif
#ifndef NPU_PREFETCH
#define NPU_PREFETCH 1
#endif
#ifndef NPU_FUSE
#define NPU_FUSE 1
#endif
/* Logical lanes are independent of host byte order. No payload contains pointers. */
typedef struct { uint32_t lane[4]; } Line128;
typedef enum { TYPE_WEIGHT, TYPE_INPUT, TYPE_BIAS, TYPE_CONFIG } TypeSel;
/* All addresses are 10-bit LINE indices, not byte addresses. */
typedef struct {
    uint32_t weight_base, bias_base, input_base, output_base, config_base;
} LayerConfig;
#ifndef POST_LATENCY
#define POST_LATENCY 1
#endif
#if POST_LATENCY != 0 && POST_LATENCY != 1
#error POST_LATENCY must be 0 or 1
#endif

/* Schedule words contain controls and addresses only. Capture enables are
 * decoded from the previous request by the Control Path at playback time. */
typedef struct {
    bool read_enable, store_en, compute_en, post_en, forward_en;
    bool we_weight, we_input, we_bias, we_cfg;
    TypeSel read_type;
    unsigned read_bank, capture_bank, forward_bank;
    uint32_t rd_addr, store_addr;
    uint16_t byte_enable;
    unsigned layer, bank, k, batch;
} Control;

typedef struct { Line128 line; uint8_t loaded; } RegValue;
typedef struct { RegValue bank[PARAM_BANKS]; } RegBanks;
/* PE coordinates = OUTPUT NEURON r, INDEPENDENT VECTOR v.
 * Reduction/input index k advances over four clocks. This 64-byte payload
 * is the registered 4x4 INT32 accumulator tile, never a combinational chain. */
typedef struct { int32_t value[4][4]; } TileSums;
typedef struct { unsigned batch; bool prefetch, fuse; } NpuOptions;
static inline int8_t unpack_i8(uint32_t word,unsigned byte) {
    int v=(int)((word>>(8*byte))&255u);
    return (int8_t)(v<128?v:v-256);
}
static inline int32_t unpack_i32(uint32_t word) {
    return word<=INT32_MAX?(int32_t)word:-1-(int32_t)(UINT32_MAX-word);
}
static inline uint32_t pack_i8(const int8_t data[4]) {
    uint32_t word=0;
    for(unsigned i=0;i<4;i++) word|=(uint32_t)(uint8_t)data[i]<<(8*i);
    return word;
}
static inline uint16_t batch_byte_enable(unsigned batch) {
    return (uint16_t)((UINT32_C(1)<<(4*batch))-1u); /* validated batch=1..4 */
}
#endif
