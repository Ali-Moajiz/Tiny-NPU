#include "scale_clamp.h"
#include "logging.h"
#include <stdio.h>
/* Exact signed nearest-even rounding, no signed shifts or overflow. Inputs
 * are validated: positive INT32 multiplier and shift in [0,62]. */
int8_t scale_clamp_value(int32_t value, int32_t multiplier, unsigned shift)
{
    int64_t product = (int64_t)value * multiplier;
    bool negative = product < 0;
    uint64_t magnitude = (uint64_t)(negative ? -product : product), rounded = magnitude;
    if (shift)
    {
        uint64_t divisor = UINT64_C(1) << shift, quotient = magnitude / divisor;
        uint64_t remainder = magnitude % divisor, half = divisor / 2;
        if (remainder > half || (remainder == half && (quotient & 1)))
            quotient++;
        rounded = quotient;
    }
    if (negative)
        return rounded >= 128 ? INT8_MIN : (int8_t)-(int64_t)rounded;
    return rounded > 127 ? INT8_MAX : (int8_t)rounded;
}
static bool evaluate(void *instance, uint64_t cycle)
{
    (void)cycle;
    ScaleClamp *s = instance;
    AccSignal a = AccSignal_read(s->ports.in.activated);
    s->evaluated = (ResultSignal){0};
    if (a.valid)
    {
        s->evaluated.valid = true;
        s->evaluated.neurons = a.neurons;
        s->evaluated.batch = a.batch;
        for (unsigned r = 0; r < a.neurons; r++)
            for (unsigned v = 0; v < a.batch; v++)
                line_set_byte(&s->evaluated.tile, v * 4 + r, (uint8_t)scale_clamp_value(a.tile.value[r][v], (int32_t)a.config.lane[0], a.config.lane[1]));
    }
    ResultSignal out = s->regs.result;
#if POST_LATENCY == 0
    if (a.valid)
        out = s->evaluated;
#endif
    ResultSignal_drive(s->ports.out.result, out);
    return true;
}
static void clock_scale(void *instance, uint64_t cycle)
{
    ScaleClamp *s = instance;
    if (logging_ports_enabled())
    {
        log_block_begin("Scale / clamp");
        printf("  in_valid=%u out_valid=%u post_latency=%u\n", AccSignal_read(s->ports.in.activated).valid, ResultSignal_read(s->ports.out.result).valid, POST_LATENCY);
        log_tile("in", AccSignal_read(s->ports.in.activated).tile);
        log_line("out", ResultSignal_read(s->ports.out.result).tile);
    }
    if (s->evaluated.valid)
    {
        s->regs.result = s->evaluated;
        log_flow(cycle, "Scale", "RESULT", "pc=0x%05X available_cycle=%llu tile=%08X_%08X_%08X_%08X",
                 Control_read(s->ports.in.control).pc, (unsigned long long)(cycle + POST_LATENCY),
                 s->evaluated.tile.lane[3], s->evaluated.tile.lane[2], s->evaluated.tile.lane[1], s->evaluated.tile.lane[0]);
    }
}
const BlockOps scale_clamp_ops = {.eval = evaluate, .clock = clock_scale};
