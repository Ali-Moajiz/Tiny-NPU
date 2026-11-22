#include "mac.h"
#include "sim.h"
#include <stdio.h>
#include <string.h>
#include "activation.h"

/* ---- PE datapath and registers ---- */

MacPeDatapath mac_pe_datapath(int8_t x, int8_t w, int32_t bias,
                             int32_t accumulator, bool seed_bias)
{
    MacPeDatapath d;
    d.x = x;
    d.w = w;
    d.product = (int16_t)((int)x * (int)w);
    d.addend = seed_bias ? bias : accumulator;
    d.wide_sum = (int64_t)d.addend + d.product;
    return d;
}

static void pe_publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    MacPe *p = instance;
    MacResult_drive(p->ports.out.sum, (MacResult){p->regs.accumulator, p->regs.valid});
}
static bool pe_evaluate(void *instance, uint64_t cycle)
{
    (void)cycle;
    MacPe *p = instance;
    Control c = Control_read(p->ports.in.control);
    p->enabled = c.compute_en && p->vector < c.batch && p->neuron < c.neurons;
    if (!p->enabled) return true; /* Clock-enable holds the accumulator. */
    p->comb = mac_pe_datapath(Operand8_read(p->ports.in.x).value,
                             Operand8_read(p->ports.in.w).value,
                             Bias32_read(p->ports.in.bias).value,
                             p->regs.accumulator, c.k == 0 && !c.accumulate);
    return true;
}
static bool pe_check(void *instance, uint64_t cycle)
{
    MacPe *p = instance;
    Control c = Control_read(p->ports.in.control);
    if (p->enabled && (c.k != 0 || c.accumulate) && !p->regs.valid) {
        log_error(cycle, "MAC PE", "feedback requires a valid accumulator neuron=%u vector=%u",
                  p->neuron, p->vector);
        return false;
    }
    if (p->enabled && (p->comb.wide_sum < INT32_MIN || p->comb.wide_sum > INT32_MAX)) {
        log_error(cycle, "MAC PE", "INT32 overflow neuron=%u vector=%u", p->neuron, p->vector);
        return false; /* Framework rejects the whole edge, including other PEs. */
    }
    return true;
}
static void pe_clock(void *instance, uint64_t cycle)
{
    MacPe *p = instance;
    Control c = Control_read(p->ports.in.control);
    if (logging_ports_enabled()) {
        log_block_begin("MAC PE");
        printf("  cell[%u][%u] enable=%u x=%d w=%d product=%d addend=%d acc=%d valid=%u\n",
               p->vector, p->neuron, p->enabled,
               Operand8_read(p->ports.in.x).value, Operand8_read(p->ports.in.w).value,
               p->enabled ? p->comb.product : 0, p->enabled ? p->comb.addend : 0,
               p->regs.accumulator, p->regs.valid);
    }
    if (p->enabled) {
        p->regs.accumulator = (int32_t)p->comb.wide_sum;
        p->regs.valid = true;
        log_flow(cycle, "MAC PE", "PE", "pc=0x%05X k=%u r=%u v=%u acc=%d x=%d w=%d product=%d prior=%d",
                 c.pc, c.k, p->neuron, p->vector, p->regs.accumulator,
                 p->comb.x, p->comb.w, p->comb.product, p->comb.addend);
    }
    if (c.post_en && c.post_col + 1 == c.neurons) p->regs.valid = false;
}
const BlockOps mac_pe_ops = {.publish=pe_publish, .eval=pe_evaluate, .check=pe_check, .clock=pe_clock};

/* ---- Array/container wiring and control ---- */

/* Operand-selection muxes and column readout; arithmetic is defined in the PE section above. */
static bool evaluate(void *instance, uint64_t cycle)
{
    (void)cycle;
    Mac *m = instance;
    Control c = Control_read(m->ports.in.control);
    const OperandSlot *s = c.slot < OPERAND_SLOTS ? &m->regs.slot[c.slot] : NULL;
    for (unsigned i = 0; i < ARRAY_SIZE; i++) {
        bool select = s && c.compute_en && c.k < ARRAY_SIZE;
        Operand8_drive(&m->wires.x[i], (Operand8){select ? line_byte(s->operand[OPERAND_X], i*4+c.k) : 0});
        Operand8_drive(&m->wires.w[i], (Operand8){select ? line_byte(s->operand[OPERAND_W], i*4+c.k) : 0});
        Bias32_drive(&m->wires.bias[i], (Bias32){s ? unpack_i32(s->operand[OPERAND_B].lane[i]) : 0});
        AccLane out = {0};
        if (c.post_en && s && c.post_col < ARRAY_SIZE && i < c.batch) {
            MacResult acc = MacResult_read(&m->wires.sum[i][c.post_col]);
            out.value = acc.value;
            out.valid = m->regs.valid && m->regs.complete && acc.valid;
            out.tag = (PostTag){.config=s->operand[OPERAND_CFG], .pc=c.pc,
                .column=c.post_col, .neurons=c.neurons, .batch=c.batch};
        }
        AccLane_drive(m->ports.out.lane[i], out);
    }
    return true;
}
static bool check(void *instance, uint64_t cycle)
{
    Mac *m = instance;
    Control c = Control_read(m->ports.in.control);
    if (c.latch_en) {
        if (c.latch_slot >= OPERAND_SLOTS || c.latch_field > OPERAND_CFG ||
            !TileSignal_read(m->ports.in.response).valid || c.compute_en || c.forward_en || c.post_en) goto fault;
        if (c.latch_field == OPERAND_CFG &&
            !activation_config_valid(TileSignal_read(m->ports.in.response).tile)) goto fault;
    }
    if (c.forward_en) {
        ResultSignal result = ResultSignal_read(m->ports.in.forward);
        if (c.forward_slot >= OPERAND_SLOTS || !result.valid || result.neurons != c.neurons ||
            result.batch != c.batch || c.compute_en || c.post_en) goto fault;
    }
    if (c.post_en) {
        if (c.slot >= OPERAND_SLOTS || !c.neurons || c.neurons > ARRAY_SIZE || !c.batch ||
            c.batch > ARRAY_SIZE || c.compute_en || !m->regs.valid || !m->regs.complete ||
            m->regs.neurons != c.neurons || m->regs.batch != c.batch ||
            c.post_col >= c.neurons || c.post_col != m->regs.post_next ||
            !m->regs.slot[c.slot].loaded[OPERAND_CFG] ||
            !activation_config_valid(m->regs.slot[c.slot].operand[OPERAND_CFG])) {
            log_error(cycle, "MAC", "POST requires matching live accumulators, ordered columns and valid CFG");
            return false;
        }
        for (unsigned v = 0; v < c.batch; v++) {
            if (!MacResult_read(&m->wires.sum[v][c.post_col]).valid) {
                log_error(cycle, "MAC", "POST readout requires a valid PE neuron=%u vector=%u", c.post_col, v);
                return false;
            }
        }
    }
    if (!c.compute_en) return true;
    if (c.slot >= OPERAND_SLOTS || !c.neurons || c.neurons > ARRAY_SIZE || !c.batch ||
        c.batch > ARRAY_SIZE || !c.k_count || c.k_count > ARRAY_SIZE || c.k >= c.k_count || c.post_en) goto fault;
    const OperandSlot *s = &m->regs.slot[c.slot];
    if (!s->loaded[OPERAND_X] || !s->loaded[OPERAND_W] ||
        (!c.accumulate && !s->loaded[OPERAND_B])) goto fault;
    if (c.k == 0) {
        if (c.accumulate && (!m->regs.valid || !m->regs.complete ||
            m->regs.neurons != c.neurons || m->regs.batch != c.batch)) goto fault;
    } else if (!m->regs.valid || m->regs.complete || m->regs.next_k != c.k ||
        m->regs.neurons != c.neurons || m->regs.batch != c.batch ||
        m->regs.compute_slot != c.slot || m->regs.k_count != c.k_count) goto fault;
    return true;
fault:
    log_error(cycle, "MAC", "invalid operands, config, accumulation or forwarding pc=0x%05X", c.pc);
    return false;
}
static void clock_mac(void *instance, uint64_t cycle)
{
    Mac *m = instance;
    Control c = Control_read(m->ports.in.control);
    if (logging_ports_enabled()) {
        log_block_begin("MAC array");
        printf("  compute=%u k=%u/%u slot=%u N=%u batch=%u retain=%u post=%u column=%u complete=%u\n",
            c.compute_en, c.k, c.k_count, c.slot, c.neurons, c.batch, c.accumulate,
            c.post_en, c.post_col, m->regs.complete);
        for (unsigned slot=0; slot<OPERAND_SLOTS; slot++) for (unsigned field=0; field<4; field++) {
            char name[48];
            snprintf(name, sizeof name, "slot%u.field%u valid=%u", slot, field, m->regs.slot[slot].loaded[field]);
            log_line(name, m->regs.slot[slot].operand[field]);
        }
    }
    if (c.latch_en) {
        OperandSlot *s = &m->regs.slot[c.latch_slot];
        s->operand[c.latch_field] = TileSignal_read(m->ports.in.response).tile;
        s->loaded[c.latch_field] = true;
        static const char *fields[] = {"X", "W", "B", "CFG"};
        log_flow(cycle, "MAC", "LATCH", "slot=%u field=%s", c.latch_slot, fields[c.latch_field]);
    }
    if (c.forward_en) {
        m->regs.slot[c.forward_slot].operand[OPERAND_X] = ResultSignal_read(m->ports.in.forward).tile;
        m->regs.slot[c.forward_slot].loaded[OPERAND_X] = true;
        log_flow(cycle, "MAC", "FORWARD", "slot=%u N=%u V=%u", c.forward_slot, c.neurons, c.batch);
    }
    if (c.compute_en) {
        m->regs.valid = true;
        m->regs.complete = c.k + 1 == c.k_count;
        m->regs.next_k = c.k + 1;
        m->regs.k_count = c.k_count;
        m->regs.compute_slot = c.slot;
        m->regs.neurons = c.neurons;
        m->regs.batch = c.batch;
        m->regs.post_next = 0;
        m->regs.compute_cycles++;
        m->regs.useful_macs += c.neurons*c.batch;
        log_flow(cycle, "MAC", "COMPUTE", "pc=0x%05X slot=%u k=%u active=%u/16 useful_MACs_this_clock=%u",
            c.pc, c.slot, c.k, c.neurons*c.batch, c.neurons*c.batch);
    }
    if (c.post_en) {
        log_flow(cycle, "MAC", "READOUT", "pc=0x%05X col=%u lanes=%u", c.pc, c.post_col, c.batch);
        m->regs.post_next++;
        if (c.post_col + 1 == c.neurons) {
            m->regs.valid = false;
            m->regs.complete = false;
            m->regs.post_next = 0;
        }
    }
}
const BlockOps mac_ops = {.eval=evaluate, .check=check, .clock=clock_mac};
