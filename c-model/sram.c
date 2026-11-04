#include "sram.h"
#include "logging.h"
#include "trained_npu_params.h"
#include <stdio.h>

static Line128 mask_write(Line128 line, Line128 data, uint16_t byte_enable)
{
    for (unsigned byte = 0; byte < 16; byte++) {
        if (byte_enable & (1u << byte)) {
            unsigned word = byte / 4, shift = 8 * (byte % 4);
            uint32_t mask = UINT32_C(255) << shift;
            line.lane[word] = (line.lane[word] & ~mask) | (data.lane[word] & mask);
        }
    }
    return line;
}

static void sram_publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    Sram *s = instance;
    TileSignal_drive(s->ports.out.response,
                     (TileSignal){s->regs.response, s->regs.response_valid});
}

static bool sram_check(void *instance, uint64_t cycle)
{
    const Sram *s = instance;
    Control c = Control_read(s->ports.in.control);
    uint32_t address = c.store_en ? c.store_addr : c.rd_addr;
    if ((c.read_enable || c.store_en) &&
        (!sram_address_matches_type(address, c.store_en ? TYPE_INPUT : c.read_type) ||
         (c.read_enable && c.store_en))) {
        log_error(cycle, "SRAM", "out-of-range line address or simultaneous read/write");
        return false;
    }
    if (c.store_en && !TileSignal_read(s->ports.in.result).valid) {
        log_error(cycle, "SRAM", "store without a valid result");
        return false;
    }
    unsigned enables = c.we_weight + c.we_input + c.we_bias + c.we_cfg;
    if (enables != (unsigned)s->regs.response_valid) {
        log_error(cycle, "SRAM", "response validity and capture enables disagree");
        return false;
    }
    return true;
}

static void sram_clock(void *instance, uint64_t cycle)
{
    Sram *s = instance;
    Control c = Control_read(s->ports.in.control);
    bool old_valid = s->regs.response_valid;
    s->regs.response_valid = c.read_enable;
    if (c.read_enable) {
        Line128 line = s->regs.lines[c.rd_addr];
        s->regs.response = line;
        s->regs.reads++;
        log_flow(cycle, "SRAM", "READ_128",
                 "line_addr=0x%03X type=%u Q=%08X_%08X_%08X_%08X valid_cycle=%llu",
                 c.rd_addr, (unsigned)c.read_type, line.lane[3], line.lane[2],
                 line.lane[1], line.lane[0], (unsigned long long)(cycle + 1));
    }
    if (c.store_en) {
        TileSignal result = TileSignal_read(s->ports.in.result);
        Line128 line = mask_write(s->regs.lines[c.store_addr], result.tile, c.byte_enable);
        s->regs.lines[c.store_addr] = line;
        s->regs.writes++;
        log_flow(cycle, "SRAM", "WRITE_128",
                 "line_addr=0x%03X BE=0x%04X line=%08X_%08X_%08X_%08X",
                 c.store_addr, c.byte_enable, line.lane[3], line.lane[2], line.lane[1], line.lane[0]);
    }
    if (logging_ports_enabled()) {
        log_block_begin("SRAM");
        printf("  in read_en=%u rd_addr=0x%03X store_en=%u store_addr=0x%03X be=0x%04X\n",
               c.read_enable, c.rd_addr, c.store_en, c.store_addr, c.byte_enable);
        log_line("out response", TileSignal_read(s->ports.out.response).tile);
        printf("  regs response_valid %u -> %u reads=%u writes=%u\n",
               old_valid, s->regs.response_valid, s->regs.reads, s->regs.writes);
    }
}

const BlockOps sram_ops = {.publish = sram_publish, .check = sram_check, .clock = sram_clock};

void sram_init(Sram *s)
{
    *s = (Sram){0};
    for (unsigned l = 0; l < NETWORK_LAYERS; l++)
    {
        Line128 *w = &s->regs.lines[SRAM_L1_WEIGHTS + l];
        Line128 *b = &s->regs.lines[SRAM_L1_BIASES + l];
        for (unsigned r = 0; r < 4; r++)
        {
            w->lane[r] = pack_i8(trained_weights[l][r]); /* lane r = row r, byte c = W[r][c] */
            b->lane[r] = (uint32_t)trained_biases[l][r];
        }
        s->regs.lines[SRAM_L1_CONFIG + l] = (Line128){{(uint32_t)requant_multiplier[l], requant_shift[l], relu_enable[l], 0}};
    }
    const int8_t initial[4][4]={{1,2,3,4},{-1,0,1,2},{10,-20,30,-40},{127,-128,64,-64}};
    for(unsigned v=0;v<4;v++) s->regs.lines[SRAM_ACTIVATION_A].lane[v]=pack_i8(initial[v]);
}
