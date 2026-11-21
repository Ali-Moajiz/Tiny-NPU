#include "scale_pe.h"
#include "pe_math.h"
#include "logging.h"
#include <stdio.h>

static int8_t clamp_i8(int64_t value)
{
    return value > INT8_MAX ? INT8_MAX : value < INT8_MIN ? INT8_MIN : (int8_t)value;
}
int8_t scale_clamp_value(int32_t value, int32_t multiplier, unsigned shift)
{
    return clamp_i8(pe_round_shift((int64_t)value * multiplier, shift));
}
static void publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    ScalePe *p = instance;
    ByteLane_drive(p->ports.out.value, p->regs.output);
}
static bool evaluate(void *instance, uint64_t cycle)
{
    ScalePe *p = instance;
    AccLane in = AccLane_read(p->ports.in.value);
    p->comb.result = (ByteLane){.valid=in.valid, .tag=in.tag};
    if (!in.valid) return true;
    uint32_t multiplier = in.tag.config.lane[0], shift = in.tag.config.lane[1];
    if (!multiplier || multiplier > INT32_MAX || shift > 62) {
        log_error(cycle, "Scale PE", "invalid multiplier / shift lane=%u", p->lane);
        return false;
    }
    p->comb.product = (int64_t)in.value * multiplier;
    p->comb.rounded = pe_round_shift(p->comb.product, shift);
    p->comb.result.value = clamp_i8(p->comb.rounded);
    return true;
}
static void clock_pe(void *instance, uint64_t cycle)
{
    ScalePe *p = instance;
    AccLane in = AccLane_read(p->ports.in.value);
    if (logging_ports_enabled()) {
        log_block_begin("Scale PE");
        printf("  lane=%u in_valid=%u in=%d M=%u S=%u out_valid=%u out=%d column=%u\n",
               p->lane, in.valid, in.value, in.tag.config.lane[0], in.tag.config.lane[1],
               p->regs.output.valid, p->regs.output.value, in.tag.column);
    }
    if (in.valid) log_flow(cycle, "Scale PE", "SCALE", "pc=0x%05X col=%u lane=%u in=%d out=%d available_cycle=%llu",
        in.tag.pc, in.tag.column, p->lane, in.value, p->comb.result.value, (unsigned long long)(cycle + 1));
    p->regs.output = p->comb.result;
}
const BlockOps scale_pe_ops = {.publish=publish, .eval=evaluate, .clock=clock_pe};
