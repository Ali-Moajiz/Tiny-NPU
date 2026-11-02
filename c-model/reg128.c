#include "reg128.h"
#include "logging.h"
/* Model-only sanity check of the Config line layout:
 * lane0=M (1..INT32_MAX), lane1=S (0..62), lane2=ReLU (0/1), lane3 reserved=0. */
static bool config_line_valid(const Line128 *q)
{
    return q->lane[0] != 0 && q->lane[0] <= INT32_MAX && q->lane[1] <= 62 && q->lane[2] <= 1 && q->lane[3] == 0;
}
void reg128Busses(void *context, uint64_t cycle)
{
    Reg128 *r = context;
    Control c;
    Line128 d;
    log_block_begin(r->name);
    if (bus_read(r->ports.control, &c, sizeof c) && c.tile_start)
        r->value.loaded = 0; /* "Tile start clears validity"; the data bits are retained */
    if (bus_read(r->ports.in, &d, sizeof d))
    {
        if (r->type == TYPE_CONFIG && !config_line_valid(&d))
        {
            r->error = true;
            log_error(cycle, "Config reg", "invalid M/S/ReLU or nonzero reserved lane");
            return;
        }
        r->value.line = d;
        r->value.loaded = 1;
        r->captures++;
        log_flow(cycle, r->name, "CAPTURE", "%08X_%08X_%08X_%08X", d.lane[3], d.lane[2], d.lane[1], d.lane[0]);
    }
    bus_write(r->ports.out, &r->value, sizeof r->value);
    log_port("D (demux out)", r->ports.in, LOG_INPUT, LOG_U128);
    log_port("Q (register)", r->ports.out, LOG_OUTPUT, LOG_REG128);
}
bool reg128_init(Reg128 *r, BusNetwork *n, const char *name, TypeSel type, const Bus *control, const Bus *in)
{
    if (!r || !name || !control || !in)
        return false;
    *r = (Reg128){0};
    r->name = name;
    r->type = type;
    r->ports.control = control;
    r->ports.in = in;
    r->ports.out = &r->out_bus;
    return bus_init(&r->out_bus, sizeof(RegValue)) && bus_register_link(n, &r->out_bus) && bus_register_module(n, reg128Busses, r);
}
