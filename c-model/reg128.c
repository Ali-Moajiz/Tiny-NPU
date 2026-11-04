#include "reg128.h"
#include "logging.h"
#include <stdio.h>

static bool config_valid(Line128 line)
{
    return line.lane[0] && line.lane[0] <= INT32_MAX &&
           line.lane[1] <= 62 && line.lane[2] <= 1 && !line.lane[3];
}

static bool write_enabled(TypeSel type, Control c)
{
    switch (type) {
    case TYPE_WEIGHT: return c.we_weight;
    case TYPE_INPUT: return c.we_input;
    case TYPE_BIAS: return c.we_bias;
    case TYPE_CONFIG: return c.we_cfg;
    }
    return false;
}

static bool reg128_eval(void *instance, uint64_t cycle)
{
    Reg128 *r = instance;
    unsigned bank = Control_read(r->ports.in.control).bank;
    if (bank >= PARAM_BANKS) {
        log_error(cycle, r->name, "invalid read bank");
        return false;
    }
    RegValue selected = r->regs.banks.bank[bank];
    TileSignal_drive(r->ports.out.selected, (TileSignal){selected.line, selected.loaded != 0});
    return true;
}

static bool reg128_check(void *instance, uint64_t cycle)
{
    const Reg128 *r = instance;
    Control c = Control_read(r->ports.in.control);
    bool capture = write_enabled(r->type, c);
    bool forward = r->type == TYPE_INPUT && c.forward_en;
    if (!capture && !forward)
        return true;
    unsigned bank = forward ? c.forward_bank : c.capture_bank;
    TileSignal data = TileSignal_read(forward ? r->ports.in.hidden : r->ports.in.response);
    bool live = bank == c.bank &&
                ((c.compute_en && r->type != TYPE_CONFIG) ||
                 (c.post_en && r->type == TYPE_CONFIG));
    if ((capture && forward) || !data.valid || bank >= PARAM_BANKS || live ||
        r->regs.banks.bank[bank].loaded ||
        (r->type == TYPE_CONFIG && !config_valid(data.tile))) {
        log_error(cycle, r->name, "invalid capture, configuration or bank collision");
        return false;
    }
    return true;
}

static void reg128_clock(void *instance, uint64_t cycle)
{
    Reg128 *r = instance;
    Control c = Control_read(r->ports.in.control);
    bool capture = write_enabled(r->type, c);
    bool forward = r->type == TYPE_INPUT && c.forward_en;
    unsigned bank = forward ? c.forward_bank : c.capture_bank;
    if (logging_ports_enabled()) {
        log_block_begin(r->name);
        printf("  in capture=%u forward=%u destination=%u\n", capture, forward, bank);
        for (unsigned i = 0; i < PARAM_BANKS; i++) {
            printf("  out bank[%u] loaded=%u\n", i, r->regs.banks.bank[i].loaded);
            log_line("  bank data", r->regs.banks.bank[i].line);
        }
    }
    if (!capture && !forward)
        return;
    Line128 line = TileSignal_read(forward ? r->ports.in.hidden : r->ports.in.response).tile;
    r->regs.banks.bank[bank] = (RegValue){line, 1};
    r->regs.captures++;
    r->regs.forwards += forward;
    log_flow(cycle, r->name, forward ? "FORWARD_HIDDEN" : "CAPTURE",
             "bank=%u line=%08X_%08X_%08X_%08X",
             bank, line.lane[3], line.lane[2], line.lane[1], line.lane[0]);
    if (logging_ports_enabled())
        log_line("  regs captured", line);
}

const BlockOps reg128_ops = {
    .eval = reg128_eval, .check = reg128_check, .clock = reg128_clock
};
