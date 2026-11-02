#include "scale_clamp.h"
#include "logging.h"
/* Values are INT32 * positive INT32, so magnitude fits safely in INT64.
 * Never right-shift a negative signed value or negate INT64_MIN.
 * Quotient/remainder implement nearest rounding, ties to EVEN, without
 * risking overflow from adding a rounding bias to the original product. */
int8_t scale_clamp_value(int32_t value, int32_t multiplier, unsigned shift)
{
    int64_t product = (int64_t)value * multiplier;
    bool negative = product < 0;
    uint64_t magnitude = (uint64_t)(negative ? -product : product);
    uint64_t rounded = magnitude;
    if (shift)
    {
        uint64_t divisor = UINT64_C(1) << shift;
        uint64_t quotient = magnitude / divisor, remainder = magnitude % divisor;
        uint64_t half = divisor / 2;
        if (remainder > half || (remainder == half && (quotient & 1)))
            quotient++;
        rounded = quotient;
    }
    if (negative)
        return rounded >= 128 ? INT8_MIN : (int8_t)-(int64_t)rounded;
    return rounded > 127 ? INT8_MAX : (int8_t)rounded;
}
void scaleClampBusses(void *context, uint64_t cycle)
{
    ScaleClamp *s = context;
    Control c;
    RowSums sums;
    RegValue cfg;
    log_block_begin("Scale / clamp x4");
    if (bus_read(s->ports.control, &c, sizeof c) && c.kind == EDGE_POST)
    {
        if (!bus_read(s->ports.config, &cfg, sizeof cfg) || !cfg.loaded)
        {
            s->error = true;
            log_error(cycle, "Scale", "POST before Config register is loaded");
            return;
        }
        int32_t multiplier = unpack_i32(cfg.line.lane[0]);
        unsigned shift = cfg.line.lane[1];
        if (multiplier <= 0 || shift > 62 || !bus_read(s->ports.activated, &sums, sizeof sums))
        {
            s->error = true;
            log_error(cycle, "Scale", "invalid quantization configuration or absent activation");
            return;
        }
        Quantized out;
        for (unsigned r = 0; r < 4; r++)
            out.row[r] = scale_clamp_value(sums.row[r], multiplier, shift);
        bus_write(s->ports.quantized, &out, sizeof out);
        log_flow(cycle, "Scale", "QUANTIZE", "M=%d S=%u rounding=nearest_ties_even q=[%d,%d,%d,%d]",
                 multiplier, shift, out.row[0], out.row[1], out.row[2], out.row[3]);
    }
    log_port("CONTROL", s->ports.control, LOG_INPUT, LOG_BYTES);
    log_port("ACTIVATED", s->ports.activated, LOG_INPUT, LOG_I32X4);
    log_port("CONFIG_REG", s->ports.config, LOG_INPUT, LOG_REG128);
    log_port("QUANTIZED", s->ports.quantized, LOG_OUTPUT, LOG_I8X4);
}
bool scale_clamp_init(ScaleClamp *s, BusNetwork *n, const Bus *ctrl, const Bus *act, const Bus *config)
{
    if (!s || !ctrl || !act || !config)
        return false;
    *s = (ScaleClamp){0};
    s->ports.control = ctrl;
    s->ports.activated = act;
    s->ports.config = config;
    s->ports.quantized = &s->output_bus;
    return bus_init_wire(&s->output_bus, sizeof(Quantized)) && bus_register_link(n, &s->output_bus) && bus_register_module(n, scaleClampBusses, s);
}
