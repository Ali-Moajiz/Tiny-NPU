#include "scale_clamp.h"
#include "logging.h"
#include <stdio.h>
#include <string.h>

/* Four scaling PEs precede this packer. All arithmetic is in scale_pe.c. */
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
