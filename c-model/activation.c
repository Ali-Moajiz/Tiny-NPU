#include "activation.h"
#include "logging.h"
void activationBusses(void *context, uint64_t cycle)
{
    Activation *a = context;
    Control c;
    RowSums sums;
    RegValue cfg;
    log_block_begin("Activation x4");
    if (bus_read(a->ports.control, &c, sizeof c) && c.kind == EDGE_POST)
    {
        if (!bus_read(a->ports.row_sums, &sums, sizeof sums))
        {
            a->error = true;
            log_error(cycle, "Activation", "POST without final registered sums");
            return;
        }
        if (!bus_read(a->ports.config, &cfg, sizeof cfg) || !cfg.loaded)
        {
            a->error = true;
            log_error(cycle, "Activation", "POST before Config register is loaded");
            return;
        }
        bool relu = cfg.line.lane[2] != 0;
        for (unsigned r = 0; r < 4; r++)
            if (relu && sums.row[r] < 0)
                sums.row[r] = 0;
        bus_write(a->ports.activated, &sums, sizeof sums);
        log_flow(cycle, "Activation", "RELU_BYPASS", "relu=%u values=[%d,%d,%d,%d]", relu, sums.row[0], sums.row[1], sums.row[2], sums.row[3]);
    }
    log_port("CONTROL", a->ports.control, LOG_INPUT, LOG_BYTES);
    log_port("ROW_SUMS", a->ports.row_sums, LOG_INPUT, LOG_I32X4);
    log_port("CONFIG_REG", a->ports.config, LOG_INPUT, LOG_REG128);
    log_port("ACTIVATED", a->ports.activated, LOG_OUTPUT, LOG_I32X4);
}

bool activation_init(Activation *a, BusNetwork *n, const Bus *ctrl, const Bus *sums, const Bus *config)
{
    if (!a || !ctrl || !sums || !config)
        return false;
    *a = (Activation){0};
    a->ports.control = ctrl;
    a->ports.row_sums = sums;
    a->ports.config = config;
    a->ports.activated = &a->output_bus;
    return bus_init_wire(&a->output_bus, sizeof(RowSums)) && bus_register_link(n, &a->output_bus) && bus_register_module(n, activationBusses, a);
}
