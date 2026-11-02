#include "controller.h"
#include "sram.h"
#include "logging.h"
#include <stdio.h>
static const char *type_name(TypeSel t)
{
    static const char *names[] = {"W(00)", "I(01)", "B(10)", "CFG(11)"};
    return names[t];
}

/* Edge map per layer (E1..E10):
 *   E1 read W   E2 read I   E3 read B   E4 read CFG
 *   E5..E8 compute C1..C4 (column wave, 4 rows in parallel)
 *   E9 POST     E10 STORE
 * Each read lands in its register one clock later (demux type register).
 * CFG is only needed at POST, so its read overlaps compute C1. */
void controllerBusses(void *context, uint64_t cycle)
{
    Controller *c = context;
    log_block_begin("Sequencer / AGU");
    if (c->done)
        return;

    LayerConfig *cfg = &c->layers[c->layer];
    Control out = {.kind = EDGE_LOAD, .layer = c->layer, .bias_enable = true, .tile_start = c->edge == 0};

    if (c->edge < 4)
    {
        /* Type order follows the truth table: W, I, B, then CFG. */
        const TypeSel types[4] = {TYPE_WEIGHT, TYPE_INPUT, TYPE_BIAS, TYPE_CONFIG};
        const uint32_t addresses[4] = {cfg->weight_base, cfg->input_base, cfg->bias_base, cfg->config_base};
        out.read_enable = true;
        out.read_type = types[c->edge];
        out.address = addresses[c->edge];
        /* Self-check of the truth table: the address must lie in the range owned by the type. */
        if (!sram_address_valid(out.address) || !sram_address_matches_type(out.address, out.read_type))
        {
            c->error = true;
            log_error(cycle, "AGU", "line address 0x%03X is outside the range of type %s", out.address, type_name(out.read_type));
            return;
        }
        log_flow(cycle, "AGU", "READ_ISSUE", "layer=%u type=%s line_addr=0x%03X", c->layer + 1, type_name(out.read_type), out.address);
    }
    else if (c->edge >= 4 && c->edge <= 7)
    {
        out.kind = EDGE_COMPUTE;
        out.wave = c->edge - 3;
    }
    else if (c->edge == 8)
        out.kind = EDGE_POST;
    else if (c->edge == 9)
    {
        uint8_t valid;
        if (!bus_read(c->ports.fifo_valid, &valid, sizeof valid) || !valid)
        {
            c->error = true;
            log_error(cycle, "AGU", "STORE without FIFO valid");
            return;
        }
        /* The result becomes the next layer's input, so it must land in the input range. */
        if (!sram_address_valid(cfg->output_base) || !sram_address_matches_type(cfg->output_base, TYPE_INPUT))
        {
            c->error = true;
            log_error(cycle, "AGU", "output line 0x%03X is outside the input range", cfg->output_base);
            return;
        }
        out.kind = EDGE_STORE;
        out.address = cfg->output_base;
        out.byte_enable = 0xFFFF; /* q replicated into all four lanes, see fifo.c */
    }

    bus_write(c->ports.control, &out, sizeof out);

    log_flow(cycle, "AGU", "EDGE", "layer=%u edge=%u read=%u wave=%u POST=%u STORE=%u",
             c->layer + 1, c->edge + 1, out.read_enable, out.wave, out.kind == EDGE_POST, out.kind == EDGE_STORE);
    log_port("CONTROL", c->ports.control, LOG_OUTPUT, LOG_BYTES);
    log_port("FIFO_VALID", c->ports.fifo_valid, LOG_INPUT, LOG_U8);

    if (logging_ports_enabled())
        printf("  RD=%u TYPE=%u ADDR=0x%03X TILE_START=%u BIAS_EN=%u BE=0x%04X\n",
               out.read_enable, (unsigned)out.read_type, out.address, out.tile_start, out.bias_enable, out.byte_enable);

    if (++c->edge == LAYER_TOTAL_EDGES)
    {
        c->edge = 0;
        if (++c->layer == NETWORK_LAYERS)
            c->done = true;
    }
}

bool controller_init(Controller *c, BusNetwork *n, const LayerConfig cfg[NETWORK_LAYERS], const Bus *valid)
{
    if (!c || !valid)
        return false;
    *c = (Controller){0};
    c->ports.control = &c->control_bus;
    c->ports.fifo_valid = valid;
    for (unsigned l = 0; l < NETWORK_LAYERS; l++)
        c->layers[l] = cfg[l];
    return bus_init_wire(&c->control_bus, sizeof(Control)) && bus_register_link(n, &c->control_bus) && bus_register_module(n, controllerBusses, c);
}
