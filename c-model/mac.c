#include "mac.h"
#include "logging.h"
#include <stdio.h>
#include <inttypes.h>

typedef struct { Control control; TileSignal weights, inputs, biases; } MacInputs;

static MacInputs mac_inputs(const MacArray *a, Control c)
{
    return (MacInputs){c,
        TileSignal_read(a->ports.in.weights),
        TileSignal_read(a->ports.in.inputs),
        TileSignal_read(a->ports.in.biases)};
}

static int64_t mac_sum(const MacArray *a, MacInputs in, unsigned r, unsigned v,
                       uint64_t cycle, bool trace)
{
    Control c = in.control;
    int8_t w = unpack_i8(in.weights.tile.lane[r], c.k);
    int8_t x = unpack_i8(in.inputs.tile.lane[v], c.k);
    int32_t acc = c.k ? a->regs.sums.value[r][v] : unpack_i32(in.biases.tile.lane[r]);
    int16_t product = (int16_t)((int)w * x);
    int64_t sum = (int64_t)acc + product;
    if (trace)
        log_flow(cycle, "MAC_array", "MAC_UPDATE",
                 "L%u C%u PE(neuron=%u,vector=%u) x=%d w=%d acc_in=%" PRId32 "%s product=%d acc_out=%" PRId32 " done=%u",
                 c.layer + 1, c.k + 1, r + 1, v + 1, x, w, acc,
                 c.k ? "" : "(bias mux)", product, (int32_t)sum, c.k == ARRAY_SIZE - 1);
    return sum;
}

static void mac_publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    MacArray *a = instance;
    AccSignal_drive(a->ports.out.sums, (AccSignal){a->regs.sums, a->regs.final_valid});
}

static bool mac_check(void *instance, uint64_t cycle)
{
    const MacArray *a = instance;
    Control c = Control_read(a->ports.in.control);
    if (c.bank >= PARAM_BANKS || c.batch < 1 || c.batch > ARRAY_SIZE ||
        (c.compute_en && (c.k >= COMPUTE_WAVES || c.k != a->regs.next_wave))) {
        log_error(cycle, "MAC_array", "invalid compute order or bank selection");
        return false;
    }
    MacInputs in = mac_inputs(a, c);
    if (!c.compute_en)
        return true;
    if (!in.weights.valid || !in.inputs.valid || !in.biases.valid) {
        log_error(cycle, "MAC_array", "unloaded operand bank");
        return false;
    }
    for (unsigned r = 0; r < ARRAY_SIZE; r++)
        for (unsigned v = 0; v < c.batch; v++) {
            int64_t sum = mac_sum(a, in, r, v, cycle, false);
            if (sum < INT32_MIN || sum > INT32_MAX) {
                log_error(cycle, "MAC_array", "INT32 overflow neuron=%u vector=%u", r+1, v+1);
                return false;
            }
        }
    return true;
}

static void mac_clock(void *instance, uint64_t cycle)
{
    MacArray *a = instance;
    Control c = Control_read(a->ports.in.control);
    MacInputs in = mac_inputs(a, c);
    if (logging_ports_enabled()) {
        log_block_begin("MAC array");
        printf("  in compute=%u bank=%u k=%u batch=%u\n", c.compute_en, c.bank, c.k, c.batch);
        log_line("  weights", in.weights.tile);
        log_line("  inputs", in.inputs.tile);
        log_line("  biases", in.biases.tile);
        log_tile("  out accumulators", a->regs.sums);
        printf("  out valid=%u\n", a->regs.final_valid);
    }
    a->regs.final_valid = false;
    if (!c.compute_en)
        return;
    /* Each PE reads its own accumulator. Operand buses remain frozen. */
    for (unsigned r = 0; r < ARRAY_SIZE; r++)
        for (unsigned v = 0; v < ARRAY_SIZE; v++)
            a->regs.sums.value[r][v] = v < c.batch ?
                (int32_t)mac_sum(a, in, r, v, cycle, true) : 0;
    a->regs.compute_cycles++;
    unsigned active = ARRAY_SIZE * c.batch;
    a->regs.updates += active;
    if (a->regs.peak_active < active)
        a->regs.peak_active = active;
    a->regs.final_valid = c.k == COMPUTE_WAVES - 1;
    a->regs.next_wave = a->regs.final_valid ? 0 : c.k + 1;
    log_flow(cycle, "MAC_array", "UTILIZATION",
             "active=%u/16 useful_MACs_this_clock=%u k=%u/4", active, active, c.k+1);
    if (logging_ports_enabled())
        log_tile("  regs updated", a->regs.sums);
}

const BlockOps mac_ops = {.publish = mac_publish, .check = mac_check, .clock = mac_clock};
