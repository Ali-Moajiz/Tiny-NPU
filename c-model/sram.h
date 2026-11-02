#ifndef SRAM_H
#define SRAM_H
#include "datapath.h"
#define SRAM_LINES 1024u /* 10-bit line address */
#define SRAM_BYTES (16u * SRAM_LINES)

/* Address ranges selected by the Type Select bits (diagram truth table). */
#define SRAM_WEIGHT_FIRST 0x000u
#define SRAM_WEIGHT_LAST 0x1FFu
#define SRAM_INPUT_FIRST 0x200u
#define SRAM_INPUT_LAST 0x2FFu
#define SRAM_BIAS_FIRST 0x300u
#define SRAM_BIAS_LAST 0x37Fu
#define SRAM_CONFIG_FIRST 0x380u /* suggested range for the extra CFG type (11) */
#define SRAM_CONFIG_LAST 0x38Fu

/* LINE addresses, one 128-bit line per tile object. */
#define SRAM_L1_WEIGHTS 0x000u
#define SRAM_L2_WEIGHTS 0x001u
#define SRAM_ACTIVATION_A 0x200u /* initial x; final q overwrites it */
#define SRAM_ACTIVATION_B 0x201u /* hidden vector */
#define SRAM_L1_BIASES 0x300u
#define SRAM_L2_BIASES 0x301u
#define SRAM_L1_CONFIG 0x380u
#define SRAM_L2_CONFIG 0x381u

typedef struct
{
    Line128 lines[SRAM_LINES];
    Bus response_bus;
    struct
    {
        const Bus *control, *write_data;
        Bus *read_response;
    } ports;
    unsigned reads, writes;
    bool error;
} Sram;

static inline bool sram_address_valid(uint32_t address)
{
    return address < SRAM_LINES;
}
/* Does this line address lie in the range that the given type select owns? */
static inline bool sram_address_matches_type(uint32_t a, TypeSel t)
{
    switch (t)
    {
    case TYPE_WEIGHT:
        return a <= SRAM_WEIGHT_LAST; /* FIRST is 0x000 */
    case TYPE_INPUT:
        return a >= SRAM_INPUT_FIRST && a <= SRAM_INPUT_LAST;
    case TYPE_BIAS:
        return a >= SRAM_BIAS_FIRST && a <= SRAM_BIAS_LAST;
    case TYPE_CONFIG:
        return a >= SRAM_CONFIG_FIRST && a <= SRAM_CONFIG_LAST;
    }
    return false;
}
bool sram_init(Sram *, BusNetwork *, const Bus *control, const Bus *write_data);
void sramBusses(void *, uint64_t);
#endif
