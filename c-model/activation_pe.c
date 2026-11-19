#include "activation_pe.h"
#include "pe_math.h"
#include "logging.h"
#include <stdio.h>

const char *activation_name(unsigned mode)
{
    static const char *names[] = {"bypass", "ReLU", "LeakyReLU", "ClippedReLU"};
    return mode < sizeof names / sizeof names[0] ? names[mode] : "invalid";
}
static bool parameter_valid(unsigned mode, uint32_t parameter)
{
    switch (mode) {
    case ACT_BYPASS: case ACT_RELU: return parameter == 0;
    case ACT_LEAKY_RELU: return parameter <= 31;
    case ACT_CLIPPED_RELU: return parameter > 0 && parameter <= INT32_MAX;
    default: return false;
    }
}
bool activation_config_valid(Line128 c)
{
    return c.lane[0] > 0 && c.lane[0] <= INT32_MAX && c.lane[1] <= 62
        && parameter_valid(c.lane[2], c.lane[3]);
}
bool activation_value(int32_t input, unsigned mode, uint32_t parameter, int32_t *output)
{
    if (!output || !parameter_valid(mode, parameter)) return false;
    /* Hardware choices: pass-through, sign mux, rounded shift, comparators.
     * No libm, floating-point activation, or implied tanh/sigmoid support. */
    switch (mode) {
    case ACT_BYPASS: *output = input; break;
    case ACT_RELU: *output = input < 0 ? 0 : input; break;
    case ACT_LEAKY_RELU:
        *output = input < 0 ? (int32_t)pe_round_shift(input, parameter) : input;
        break;
    case ACT_CLIPPED_RELU:
        *output = input < 0 ? 0 : input > (int32_t)parameter ? (int32_t)parameter : input;
        break;
    default: return false;
    }
    return true;
}
static void publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    ActivationPe *p = instance;
    AccLane_drive(p->ports.out.value, p->regs.output);
}
static bool evaluate(void *instance, uint64_t cycle)
{
    ActivationPe *p = instance;
    p->comb.result = AccLane_read(p->ports.in.value);
    AccLane *a = &p->comb.result;
    if (a->valid && (!activation_config_valid(a->tag.config) ||
        !activation_value(a->value, a->tag.config.lane[2], a->tag.config.lane[3], &a->value))) {
        log_error(cycle, "Activation PE", "invalid activation / scale configuration lane=%u", p->lane);
        return false;
    }
    return true;
}
static void clock_pe(void *instance, uint64_t cycle)
{
    ActivationPe *p = instance;
    AccLane in = AccLane_read(p->ports.in.value);
    if (logging_ports_enabled()) {
        log_block_begin("Activation PE");
        printf("  lane=%u in_valid=%u in=%d out_valid=%u out=%d column=%u mode=%s\n",
               p->lane, in.valid, in.value, p->regs.output.valid, p->regs.output.value,
               in.tag.column, activation_name(in.tag.config.lane[2]));
    }
    if (in.valid) log_flow(cycle, "Activation PE", "ACT", "pc=0x%05X col=%u lane=%u in=%d out=%d mode=%s available_cycle=%llu",
        in.tag.pc, in.tag.column, p->lane, in.value, p->comb.result.value,
        activation_name(in.tag.config.lane[2]), (unsigned long long)(cycle + 1));
    p->regs.output = p->comb.result; /* Bubble clears valid, not a repeated token. */
}
const BlockOps activation_pe_ops = {.publish=publish, .eval=evaluate, .clock=clock_pe};
