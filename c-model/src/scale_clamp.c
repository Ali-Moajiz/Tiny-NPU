#include "scale_clamp.h"
#include "sim.h"
#include <stdio.h>
#include <string.h>

/* ---- PE datapath and registers ---- */

static int8_t clamp_i8(int64_t value)
{
    return value > INT8_MAX ? INT8_MAX : value < INT8_MIN ? INT8_MIN : (int8_t)value;
}
int8_t scale_clamp_value(int32_t value, int32_t multiplier, unsigned shift)
{
    return clamp_i8(pe_round_shift((int64_t)value * multiplier, shift));
}
static void pe_publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    ScalePe *p = instance;
    ByteLane_drive(p->ports.out.value, p->regs.output);
}
static bool pe_evaluate(void *instance, uint64_t cycle)
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
static void pe_clock(void *instance, uint64_t cycle)
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
const BlockOps scale_pe_ops = {.publish=pe_publish, .eval=pe_evaluate, .clock=pe_clock};

/* ---- Array/container wiring and control ---- */

/* Four scaling PEs precede this packer. Arithmetic is defined in the PE section above. */
static void publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    ScaleClamp *s = instance;
    ResultSignal_drive(s->ports.out.result, s->regs.result);
}
static bool evaluate(void *instance, uint64_t cycle)
{
    (void)cycle;
    ScaleClamp *s = instance;
    s->comb.capture = false;
    for (unsigned v=0; v<ARRAY_SIZE; v++) {
        s->comb.lane[v] = ByteLane_read(&s->scaled[v]);
        s->comb.capture |= s->comb.lane[v].valid;
    }
    return true;
}
static bool check(void *instance, uint64_t cycle)
{
    ScaleClamp *s = instance;
    Control c = Control_read(s->ports.in.control);
    if (c.post_en && c.post_col == 0 && (s->regs.collecting || s->comb.capture)) goto fault;
    if (!s->comb.capture) return true;
    ByteLane first = s->comb.lane[0];
    PostTag t = first.tag;
    if (!first.valid || !s->regs.collecting || t.pc != s->regs.pc ||
        t.neurons != s->regs.neurons || t.batch != s->regs.batch ||
        t.column != s->regs.next_column || t.column >= t.neurons) goto fault;
    if (t.column && memcmp(&t.config, &s->regs.config, sizeof t.config)) goto fault;
    for (unsigned v=0; v<ARRAY_SIZE; v++) {
        ByteLane a = s->comb.lane[v];
        if (a.valid != (v < t.batch)) goto fault;
        if (a.valid && (a.tag.pc != t.pc || a.tag.column != t.column ||
            a.tag.batch != t.batch || a.tag.neurons != t.neurons ||
            memcmp(&a.tag.config, &t.config, sizeof t.config))) goto fault;
    }
    return true;
fault:
    log_error(cycle, "Result packer", "invalid column order, lane mask or pipeline context");
    return false;
}
static void clock_pack(void *instance, uint64_t cycle)
{
    ScaleClamp *s = instance;
    Control c = Control_read(s->ports.in.control);
    if (logging_ports_enabled()) {
        log_block_begin("Result packer");
        printf("  collecting=%u next_column=%u capture=%u result_valid=%u\n",
            s->regs.collecting, s->regs.next_column, s->comb.capture, s->regs.result.valid);
        log_line("result", s->regs.result.tile);
    }
    if (c.post_en && c.post_col == 0) {
        s->regs.result = (ResultSignal){.neurons=c.neurons, .batch=c.batch};
        s->regs.collecting = true;
        s->regs.next_column = 0;
        s->regs.neurons = c.neurons;
        s->regs.batch = c.batch;
        s->regs.pc = c.pc;
    }
    if (!s->comb.capture) return;
    PostTag t = s->comb.lane[0].tag;
    s->regs.config = t.config;
    for (unsigned v=0; v<t.batch; v++)
        line_set_byte(&s->regs.result.tile, 4*v+t.column, (uint8_t)s->comb.lane[v].value);
    s->regs.next_column++;
    log_flow(cycle, "Result packer", "PACK", "pc=0x%05X col=%u bytes=%u layout=4*vector+neuron",
        t.pc, t.column, t.batch);
    if (s->regs.next_column == t.neurons) {
        s->regs.collecting = false;
        s->regs.result.valid = true;
        log_flow(cycle, "Scale", "RESULT", "pc=0x%05X available_cycle=%llu tile=%08X_%08X_%08X_%08X",
            t.pc, (unsigned long long)(cycle + 1), s->regs.result.tile.lane[3],
            s->regs.result.tile.lane[2], s->regs.result.tile.lane[1], s->regs.result.tile.lane[0]);
    }
}
const BlockOps scale_clamp_ops = {.publish=publish, .eval=evaluate, .check=check, .clock=clock_pack};
