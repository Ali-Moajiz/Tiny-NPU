/* Top-level netlist. This is the only place that connects module ports. */
#include "npu.h"
#include "logging.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
void npu_init(Npu *n)
{
    memset(n, 0, sizeof *n);
    controller_init(&n->controller);
    n->stream.regs.output_enabled = true;
    Control_connect(&n->wires.control, &n->bus_context, "controller.control", &n->controller);
    TileSignal_connect(&n->wires.memory, &n->bus_context, "sram.response", &n->sram);
    TileSignal_connect(&n->wires.input, &n->bus_context, "stream.input", &n->stream);
    StreamStatus_connect(&n->wires.stream_status, &n->bus_context, "stream.status", &n->stream);
    ResultSignal_connect(&n->wires.result, &n->bus_context, "scale.result", &n->scale);
    n->controller.ports.in.memory = &n->wires.memory;
    n->controller.ports.in.stream = &n->wires.stream_status;
    n->controller.ports.out.control = &n->wires.control;
    n->sram.ports.in.control = &n->wires.control;
    n->sram.ports.in.input = &n->wires.input;
    n->sram.ports.in.result = &n->wires.result;
    n->sram.ports.out.response = &n->wires.memory;
    n->stream.ports.in.control = &n->wires.control;
    n->stream.ports.in.response = &n->wires.memory;
    n->stream.ports.out.input = &n->wires.input;
    n->stream.ports.out.status = &n->wires.stream_status;
    n->mac.ports.in.control = &n->wires.control;
    n->mac.ports.in.response = &n->wires.memory;
    n->mac.ports.in.forward = &n->wires.result;
    n->scale.ports.in.control = &n->wires.control;
    n->scale.ports.out.result = &n->wires.result;

    /* Row X and column W are shared wires, not PE-to-PE forwarding. */
    for (unsigned i = 0; i < ARRAY_SIZE; i++) {
        Operand8_connect(&n->mac.wires.x[i], &n->bus_context, "mac.row_x", &n->mac);
        Operand8_connect(&n->mac.wires.w[i], &n->bus_context, "mac.column_w", &n->mac);
        Bias32_connect(&n->mac.wires.bias[i], &n->bus_context, "mac.column_bias", &n->mac);
        AccLane_connect(&n->wires.sums[i], &n->bus_context, "mac.column_readout", &n->mac);
        AccLane_connect(&n->wires.activated[i], &n->bus_context, "activation.output", &n->activation.lane[i]);
        ByteLane_connect(&n->scale.scaled[i], &n->bus_context, "scale.output", &n->scale.lane[i]);
        n->mac.ports.out.lane[i] = &n->wires.sums[i];
        n->activation.ports.in.lane[i] = &n->wires.sums[i];
        n->activation.ports.out.lane[i] = &n->wires.activated[i];
        n->scale.ports.in.lane[i] = &n->wires.activated[i];
        n->scale.lane[i].lane = i;
        n->scale.lane[i].ports.in.value = n->scale.ports.in.lane[i];
        n->scale.lane[i].ports.out.value = &n->scale.scaled[i];
        for (unsigned neuron = 0; neuron < ARRAY_SIZE; neuron++) {
            MacPe *pe = &n->mac.cell[i][neuron];
            pe->vector = i;
            pe->neuron = neuron;
            pe->ports.in.x = &n->mac.wires.x[i];
            pe->ports.in.w = &n->mac.wires.w[neuron];
            pe->ports.in.bias = &n->mac.wires.bias[neuron];
            pe->ports.in.control = &n->wires.control;
            pe->ports.out.sum = &n->mac.wires.sum[i][neuron];
            MacResult_connect(pe->ports.out.sum, &n->bus_context, "mac.pe.accumulator", pe);
        }
    }
    activation_init(&n->activation);
    /* EVAL follows combinational dependencies. Registered outputs from ALL
     * blocks are published first; CHECK completes before any clock update. */
#define REGISTER(instance, ops) \
    do { if (!framework_register(n->blocks, &n->block_count, NPU_BLOCK_CAPACITY, \
                                 (instance), (ops))) n->error = true; } while (0)
    REGISTER(&n->controller, &controller_ops);
    REGISTER(&n->sram, &sram_ops);
    REGISTER(&n->stream, &stream_ops);
    REGISTER(&n->mac, &mac_ops);
    for (unsigned r = 0; r < ARRAY_SIZE; r++)
        for (unsigned v = 0; v < ARRAY_SIZE; v++)
            REGISTER(&n->mac.cell[v][r], &mac_pe_ops);
    for (unsigned i = 0; i < ARRAY_SIZE; i++) {
        REGISTER(&n->activation.lane[i], &activation_pe_ops);
        REGISTER(&n->scale.lane[i], &scale_pe_ops);
    }
    REGISTER(&n->scale, &scale_clamp_ops);
#undef REGISTER
}
bool npu_step(Npu *n)
{
    if (n->error || n->controller.regs.done)
        return false;
    n->cycle++;
    log_cycle_begin(n->cycle);
    if (!framework_step(&n->bus_context, n->blocks, n->block_count, n->cycle))
        n->error = true;
    return !n->error;
}
bool npu_run(Npu *n, uint64_t max_cycles)
{
    while (!n->controller.regs.done && !n->error && n->cycle < max_cycles)
        (void)npu_step(n);
    if (!n->error && !n->controller.regs.done)
    {
        log_error(n->cycle, "NPU", "cycle limit: pc=0x%05X (missing input, back-pressure, loop or missing HALT)", n->controller.regs.pc);
        n->error = true;
    }
    return !n->error;
}
void npu_print_metrics(const Npu *n)
{
    double compute = n->mac.regs.compute_cycles ? 100.0 * (double)n->mac.regs.useful_macs / (16.0 * (double)n->mac.regs.compute_cycles) : 0;
    double whole = n->cycle ? 100.0 * (double)n->mac.regs.useful_macs / (16.0 * (double)n->cycle) : 0;
    printf("METRICS cycles=%" PRIu64 " instructions=%" PRIu64 " compute_cycles=%" PRIu64 " useful_macs=%" PRIu64 " bus_stall_cycles=%" PRIu64 " sram_reads=%" PRIu64 " sram_writes=%" PRIu64 " compute_util=%.2f%% whole_util=%.2f%%\n",
           n->cycle, n->controller.regs.instructions, n->mac.regs.compute_cycles, n->mac.regs.useful_macs, n->controller.regs.bus_stall_cycles, n->sram.regs.reads, n->sram.regs.writes, compute, whole);
}
