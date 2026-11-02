#include "demux.h"
#include "logging.h"
static const char *dest_name[4] = {"Weight reg", "Input reg", "Bias reg", "Config reg"};
void demuxBusses(void *context, uint64_t cycle)
{
    Demux *d = context;
    Control c;
    log_block_begin("Data steering demux 1:4 + type register");

    /* 1. Steer Q (issued one clock ago) using the OLD type register. */
    if (d->valid_q)
    {
        Line128 q;
        if (!bus_read(d->ports.sram_q, &q, sizeof q))
        {
            d->error = true;
            log_error(cycle, "Demux", "type register valid but no SRAM Q");
            return;
        }
        
        bus_write(d->ports.outputs[d->type_q], &q, sizeof q);
        d->routed[d->type_q]++;
        log_flow(cycle, "Demux", "ROUTE", "sel=%u%u -> %s Q=%08X_%08X_%08X_%08X",
                 (unsigned)d->type_q >> 1, (unsigned)d->type_q & 1u, dest_name[d->type_q], q.lane[3], q.lane[2], q.lane[1], q.lane[0]);
    }

    /* 2. Latch the type of the read issued this clock (it steers next clock). */
    if (bus_read(d->ports.control, &c, sizeof c) && c.read_enable)
    {
        d->type_q = c.read_type;
        d->valid_q = true;
    }
    else
        d->valid_q = false;

    log_port("CONTROL", d->ports.control, LOG_INPUT, LOG_BYTES);
    log_port("SRAM_Q[127:0]", d->ports.sram_q, LOG_INPUT, LOG_U128);
    log_port("OUT_W (sel=00)", d->ports.outputs[TYPE_WEIGHT], LOG_OUTPUT, LOG_U128);
    log_port("OUT_I (sel=01)", d->ports.outputs[TYPE_INPUT], LOG_OUTPUT, LOG_U128);
    log_port("OUT_B (sel=10)", d->ports.outputs[TYPE_BIAS], LOG_OUTPUT, LOG_U128);
    log_port("OUT_CFG (sel=11)", d->ports.outputs[TYPE_CONFIG], LOG_OUTPUT, LOG_U128);
}

bool demux_init(Demux *d, BusNetwork *n, const Bus *control, const Bus *sram_q)
{
    if (!d || !control || !sram_q)
        return false;
    *d = (Demux){0};
    d->ports.control = control;
    d->ports.sram_q = sram_q;
    for (unsigned i = 0; i < 4; i++)
    {
        d->ports.outputs[i] = &d->out[i];
        if (!bus_init_wire(&d->out[i], sizeof(Line128)) || !bus_register_link(n, &d->out[i]))
            return false;
    }
    return bus_register_module(n, demuxBusses, d);
}
