#include "demux.h"
#include "logging.h"
void demuxBusses(void *context, uint64_t cycle)
{
    Demux *d = context;
    Control c;
    log_block_begin("Steering demux + type/bank tag register");
    if (d->valid_q)
    {
        TaggedLine q = {.bank = d->bank_q};
        if ((unsigned)d->type_q > TYPE_CONFIG || d->bank_q >= PARAM_BANKS || !bus_read(d->ports.sram_q, &q.line, sizeof q.line))
        {
            d->error = true;
            log_error(cycle, "Demux", "invalid tag or missing SRAM response");
            return;
        }
        bus_write(d->ports.outputs[d->type_q], &q, sizeof q);
        d->routed[d->type_q]++;
        log_flow(cycle, "Demux", "ROUTE", "type=%u dest_bank=%u Q=%08X_%08X_%08X_%08X",
                 (unsigned)d->type_q, q.bank, q.line.lane[3], q.line.lane[2], q.line.lane[1], q.line.lane[0]);
    }
    /* Only AFTER routing the old Q may we latch this clock's new request tag. */
    d->valid_q = bus_read(d->ports.control, &c, sizeof c) && c.read_enable;
    if (d->valid_q)
    {
        d->type_q = c.read_type;
        d->bank_q = c.read_bank;
    }
    log_port("CONTROL", d->ports.control, LOG_INPUT, LOG_BYTES);
    log_port("SRAM_Q", d->ports.sram_q, LOG_INPUT, LOG_U128);
    for (unsigned i = 0; i < 4; i++)
        log_port("TAGGED_DESTINATION", d->ports.outputs[i], LOG_OUTPUT, LOG_BYTES);
}

bool demux_init(Demux *d, BusNetwork *n, const Bus *control, const Bus *q)
{
    if (!d || !control || !q)
        return false;
    *d = (Demux){0};
    d->ports.control = control;
    d->ports.sram_q = q;
    for (unsigned i = 0; i < 4; i++)
    {
        d->ports.outputs[i] = &d->out[i];
        if (!bus_init_wire(&d->out[i], sizeof(TaggedLine)) || !bus_register_link(n, &d->out[i]))
            return false;
    }
    return bus_register_module(n, demuxBusses, d);
}
