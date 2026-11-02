#include "sram.h"
#include "logging.h"
#include "trained_npu_params.h"
void sramBusses(void *context, uint64_t cycle)
{
    Sram *s = context;
    Control c;
    log_block_begin("SRAM 1024 x 128-bit");
    if (bus_read(s->ports.control, &c, sizeof c) && (c.read_enable || c.kind == EDGE_STORE))
    {
        if (!sram_address_valid(c.address) || (c.read_enable && c.kind == EDGE_STORE))
        {
            s->error = true;
            log_error(cycle, "SRAM", "out-of-range line address or simultaneous read/write");
            return;
        }
        Line128 *line = &s->lines[c.address];
        if (c.read_enable)
        {
            bus_write(s->ports.read_response, line, sizeof *line);
            s->reads++;
            log_flow(cycle, "SRAM", "READ_128", "line_addr=0x%03X type=%u Q=%08X_%08X_%08X_%08X valid_cycle=%llu",
                     c.address, (unsigned)c.read_type, line->lane[3], line->lane[2], line->lane[1], line->lane[0], (unsigned long long)(cycle + 1));
        }
        else
        {
            Line128 data;
            if (!bus_read(s->ports.write_data, &data, sizeof data))
            {
                s->error = true;
                log_error(cycle, "SRAM", "write without FIFO payload");
                return;
            }
            for (unsigned byte = 0; byte < 16; byte++)
                if (c.byte_enable & (1u << byte))
                {
                    unsigned word = byte / 4, shift = 8 * (byte % 4);
                    uint32_t mask = UINT32_C(255) << shift;
                    line->lane[word] = (line->lane[word] & ~mask) | (data.lane[word] & mask);
                }
            s->writes++;
            log_flow(cycle, "SRAM", "WRITE_128", "line_addr=0x%03X BE=0x%04X line=%08X_%08X_%08X_%08X",
                     c.address, c.byte_enable, line->lane[3], line->lane[2], line->lane[1], line->lane[0]);
        }
    }
    log_port("CONTROL", s->ports.control, LOG_INPUT, LOG_BYTES);
    log_port("WDATA[127:0]", s->ports.write_data, LOG_INPUT, LOG_U128);
    log_port("Q[127:0]", s->ports.read_response, LOG_OUTPUT, LOG_U128);
}

bool sram_init(Sram *s, BusNetwork *n, const Bus *ctrl, const Bus *data)
{
    *s = (Sram){0};
    s->ports.control = ctrl;
    s->ports.write_data = data;
    s->ports.read_response = &s->response_bus;
    for (unsigned l = 0; l < NETWORK_LAYERS; l++)
    {
        Line128 *w = &s->lines[SRAM_L1_WEIGHTS + l];
        Line128 *b = &s->lines[SRAM_L1_BIASES + l];
        for (unsigned r = 0; r < 4; r++)
        {
            w->lane[r] = pack_i8(trained_weights[l][r]); /* lane r = row r, byte c = W[r][c] */
            b->lane[r] = (uint32_t)trained_biases[l][r];
        }
        s->lines[SRAM_L1_CONFIG + l] = (Line128){{(uint32_t)requant_multiplier[l], requant_shift[l], relu_enable[l], 0}};
    }
    const int8_t initial[4] = {1, 2, 3, 4};
    s->lines[SRAM_ACTIVATION_A] = line_broadcast(pack_i8(initial));
    return bus_init(&s->response_bus, sizeof(Line128)) && bus_register_link(n, &s->response_bus) && bus_register_module(n, sramBusses, s);
}
