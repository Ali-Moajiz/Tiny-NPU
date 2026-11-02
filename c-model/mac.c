#include "mac.h"
#include "logging.h"
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
/* Column wave: on compute edge C_k (k=1..4) the four cells of column k fire together.
 * Cell(r,c) = psum_in + W[r][c]*I[r][c], where psum_in is the OLD psum of the cell to
 * its left, or for column 1 the output of the bias mux. */
static void macArrayProcess(MacArray *a, const Control *ctrl, uint64_t cycle)
{
    RegValue w, x, b;
    if (ctrl->wave < 1 || ctrl->wave > COMPUTE_WAVES ||
        !bus_read(a->ports.weights, &w, sizeof w) || !bus_read(a->ports.inputs, &x, sizeof x) || !bus_read(a->ports.biases, &b, sizeof b) ||
        !w.loaded || !x.loaded || !b.loaded)
    {
        a->error = true;
        log_error(cycle, "MAC_array", "compute before W, I and Bias registers are loaded");
        return;
    }
    /* Snapshot ALL old registers before ANY cell updates. This is the edge. */
    Mac old[4][4];
    memcpy(old, a->cell, sizeof old);
    unsigned c = ctrl->wave - 1;
    for (unsigned r = 0; r < 4; r++)
    {
        if (c && !old[r][c - 1].valid)
        {
            a->error = true;
            log_error(cycle, "MAC_array", "psum from left neighbour not valid");
            return;
        }
        int8_t weight = unpack_i8(w.line.lane[r], c), input = unpack_i8(x.line.lane[r], c);
        /* 2:1 bias mux exists only in front of column 1. */
        int32_t incoming = c ? old[r][c - 1].psum : (ctrl->bias_enable ? unpack_i32(b.line.lane[r]) : 0);
        int16_t product = (int16_t)((int)input * (int)weight);
        int64_t sum = (int64_t)incoming + product;
        /* Diagram forbids accumulator saturation. Reject unsupported overflow. */
        if (sum < INT32_MIN || sum > INT32_MAX)
        {
            a->error = true;
            log_error(cycle, "MAC_array", "INT32 overflow row=%u col=%u", r + 1, c + 1);
            return;
        }
        a->cell[r][c].psum = (int32_t)sum;
        a->cell[r][c].valid = true;
        a->updates++;
        if (c == 3)
            a->sums.row[r] = (int32_t)sum;
        log_flow(cycle, "MAC_array", "MAC_UPDATE", "C%u MAC%u%u I=%d w=%d psum_in=%" PRId32 "%s product=%d psum_out=%" PRId32 " row_done=%u",
                 ctrl->wave, r + 1, c + 1, (int)input, (int)weight, incoming, c ? "" : "(bias mux)", (int)product, (int32_t)sum, c == 3);
    }
    bus_write(a->ports.row_sums, &a->sums, sizeof a->sums);
}
void macArrayBusses(void *context, uint64_t cycle)
{
    MacArray *a = context;
    Control c;
    log_block_begin("MAC array (16 cells)");
    if (bus_read(a->ports.control, &c, sizeof c))
    {
        if (c.tile_start)
        {
            for (unsigned r = 0; r < 4; r++)
                for (unsigned col = 0; col < 4; col++)
                    a->cell[r][col].valid = false;
        }
        if (c.kind == EDGE_COMPUTE)
            macArrayProcess(a, &c, cycle);
    }
    log_port("CONTROL", a->ports.control, LOG_INPUT, LOG_BYTES);
    log_port("W_REG[127:0]", a->ports.weights, LOG_INPUT, LOG_REG128);
    log_port("I_REG[127:0]", a->ports.inputs, LOG_INPUT, LOG_REG128);
    log_port("BIAS_REG[127:0]", a->ports.biases, LOG_INPUT, LOG_REG128);
    log_port("ROW_SUMS", a->ports.row_sums, LOG_OUTPUT, LOG_I32X4);
    if (logging_ports_enabled())
        for (unsigned r = 0; r < 4; r++)
        {
            printf("  ROW%u", r + 1);
            for (unsigned col = 0; col < 4; col++)
            {
                Mac *m = &a->cell[r][col];
                printf(" | MAC%u%u psum=%" PRId32 " valid=%u", r + 1, col + 1, m->psum, m->valid);
            }
            putchar('\n');
        }
}
bool mac_array_init(MacArray *a, BusNetwork *n, const Bus *ctrl, const Bus *w, const Bus *x, const Bus *b)
{
    if (!a || !ctrl || !w || !x || !b)
        return false;
    *a = (MacArray){0};
    a->ports.control = ctrl;
    a->ports.weights = w;
    a->ports.inputs = x;
    a->ports.biases = b;
    a->ports.row_sums = &a->sums_bus;
    return bus_init(&a->sums_bus, sizeof(RowSums)) && bus_register_link(n, &a->sums_bus) && bus_register_module(n, macArrayBusses, a);
}
