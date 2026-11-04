#ifndef SRAM_H
#define SRAM_H
#include "signals.h"
#include "framework.h"
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
#define SRAM_ACTIVATION_A 0x200u /* up to four independent input vectors; final outputs overwrite active lanes */
#define SRAM_ACTIVATION_B 0x201u /* hidden tile; untouched with fusion enabled */
#define SRAM_L1_BIASES 0x300u
#define SRAM_L2_BIASES 0x301u
#define SRAM_L1_CONFIG 0x380u
#define SRAM_L2_CONFIG 0x381u

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
typedef struct
{
    struct
    {
        const ControlBus *control;
        const TileSignalBus *result;
    } in;
    struct
    {
        TileSignalBus *response;
    } out;
} SramPorts;

typedef struct
{
    Line128 lines[SRAM_LINES];
    Line128 response;
    bool response_valid;
    unsigned reads, writes;
} SramRegs;

typedef struct
{
    SramPorts ports;
    SramRegs regs;
} Sram;

void sram_init(Sram *);
extern const BlockOps sram_ops;
#endif
