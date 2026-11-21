#include "mac_pe.h"
#include "logging.h"
#include <stdio.h>

static void publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    MacPe *p = instance;
    MacResult_drive(p->ports.out.sum, (MacResult){p->regs.accumulator, p->regs.valid});
}
static bool evaluate(void *instance, uint64_t cycle)
{
    (void)cycle;
    MacPe *p = instance;
    Control c = Control_read(p->ports.in.control);
    p->comb.enabled = c.compute_en && p->vector < c.batch && p->neuron < c.neurons;
    if (!p->comb.enabled) return true; /* Clock-enable holds the accumulator. */
    p->comb.x = Operand8_read(p->ports.in.x).value;
    p->comb.w = Operand8_read(p->ports.in.w).value;
    p->comb.product = (int16_t)((int)p->comb.x * (int)p->comb.w);
    p->comb.addend = c.k == 0 && !c.accumulate
        ? Bias32_read(p->ports.in.bias).value : p->regs.accumulator;
    p->comb.wide_sum = (int64_t)p->comb.addend + p->comb.product;
    return true;
}
static bool check(void *instance, uint64_t cycle)
{
    MacPe *p = instance;
    Control c = Control_read(p->ports.in.control);
    if (p->comb.enabled && (c.k != 0 || c.accumulate) && !p->regs.valid) {
        log_error(cycle, "MAC PE", "feedback requires a valid accumulator neuron=%u vector=%u",
                  p->neuron, p->vector);
        return false;
    }
    if (p->comb.enabled && (p->comb.wide_sum < INT32_MIN || p->comb.wide_sum > INT32_MAX)) {
        log_error(cycle, "MAC PE", "INT32 overflow neuron=%u vector=%u", p->neuron, p->vector);
        return false; /* Framework rejects the whole edge, including other PEs. */
    }
    return true;
}
static void clock_pe(void *instance, uint64_t cycle)
{
    MacPe *p = instance;
    Control c = Control_read(p->ports.in.control);
    if (logging_ports_enabled()) {
        log_block_begin("MAC PE");
        printf("  cell[%u][%u] enable=%u x=%d w=%d product=%d addend=%d acc=%d valid=%u\n",
               p->vector, p->neuron, p->comb.enabled,
               Operand8_read(p->ports.in.x).value, Operand8_read(p->ports.in.w).value,
               p->comb.enabled ? p->comb.product : 0, p->comb.enabled ? p->comb.addend : 0,
               p->regs.accumulator, p->regs.valid);
    }
    if (p->comb.enabled) {
        p->regs.accumulator = (int32_t)p->comb.wide_sum;
        p->regs.valid = true;
        log_flow(cycle, "MAC PE", "PE", "pc=0x%05X k=%u r=%u v=%u acc=%d x=%d w=%d product=%d prior=%d",
                 c.pc, c.k, p->neuron, p->vector, p->regs.accumulator,
                 p->comb.x, p->comb.w, p->comb.product, p->comb.addend);
    }
    if (c.post_en && c.post_col + 1 == c.neurons) p->regs.valid = false;
}
const BlockOps mac_pe_ops = {.publish=publish, .eval=evaluate, .check=check, .clock=clock_pe};
