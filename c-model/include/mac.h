#ifndef MAC_H
#define MAC_H
#include "types.h"

/* ---- One processing element ---- */

/* One physical multiplier/accumulator cell. Operand registers live in its
 * owning MAC block; shared row/column buses feed this combinational datapath. */
typedef struct { int32_t accumulator; bool valid; } MacPeRegisters;
/* Named combinational wires. This scalar datapath has no bus/host dependency. */
typedef struct {
    int8_t x, w;
    int16_t product;
    int32_t addend;
    int64_t wide_sum; /* C guard width; accumulator storage stays INT32. */
} MacPeDatapath;
MacPeDatapath mac_pe_datapath(int8_t x, int8_t w, int32_t bias,
                             int32_t accumulator, bool seed_bias);
typedef struct
{
    struct {
        const Operand8Bus *x, *w;
        const Bias32Bus *bias;
        const ControlBus *control;
    } in;
    struct { MacResultBus *sum; } out;
} MacPePorts;
typedef struct
{
    MacPeRegisters regs;
    MacPePorts ports;
    unsigned vector, neuron; /* Static instance coordinates. */
    bool enabled; /* Combinational clock-enable, not another register. */
    MacPeDatapath comb;
} MacPe;
extern const BlockOps mac_pe_ops;

/* ---- Array/container ---- */

typedef struct { Line128 operand[4]; bool loaded[4]; } OperandSlot;
typedef struct {
    OperandSlot slot[OPERAND_SLOTS];
    bool valid, complete;
    unsigned neurons, batch, next_k, k_count, compute_slot, post_next;
    uint64_t useful_macs, compute_cycles;
} MacRegisters;
typedef struct {
    struct {
        const ControlBus *control;
        const TileSignalBus *response;
        const ResultSignalBus *forward;
    } in;
    struct { AccLaneBus *lane[ARRAY_SIZE]; } out;
} MacPorts;
typedef struct {
    MacRegisters regs;
    MacPorts ports;
    MacPe cell[ARRAY_SIZE][ARRAY_SIZE]; /* [vector][neuron] */
    struct {
        Operand8Bus x[ARRAY_SIZE], w[ARRAY_SIZE];
        Bias32Bus bias[ARRAY_SIZE];
        MacResultBus sum[ARRAY_SIZE][ARRAY_SIZE];
    } wires;
} Mac;
extern const BlockOps mac_ops;

#endif
