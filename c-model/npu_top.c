#include "npu.h"
#include "logging.h"

/* Static netlist. Connections name both the receiving pin and its bus. */
static void wire_blocks(Npu *n)
{
    DataPath *d = &n->data;
    NpuBuses *b = &n->buses;
    BusContext *ctx = &n->bus_context;

    Control_connect(&b->control, ctx, "control", &n->controller);
    TileSignal_connect(&b->sram_response, ctx, "sram_response", &d->sram);
    TileSignal_connect(&b->weights, ctx, "weights", &d->reg_w);
    TileSignal_connect(&b->inputs, ctx, "inputs", &d->reg_i);
    TileSignal_connect(&b->biases, ctx, "biases", &d->reg_b);
    TileSignal_connect(&b->config, ctx, "config", &d->reg_cfg);
    TileSignal_connect(&b->post_result, ctx, "post_result", &d->post);
    AccSignal_connect(&b->mac_sums, ctx, "mac_sums", &d->array);

    n->controller.ports = (ControllerPorts){.out.control = &b->control};
    d->sram.ports = (SramPorts){
        .in = {.control = &b->control, .result = &b->post_result}, .out.response = &b->sram_response};
    d->reg_w.ports = (Reg128Ports){
        .in = {.control = &b->control, .response = &b->sram_response}, .out.selected = &b->weights};
    d->reg_i.ports = (Reg128Ports){
        .in = {.control = &b->control, .response = &b->sram_response, .hidden = &b->post_result}, .out.selected = &b->inputs};
    d->reg_b.ports = (Reg128Ports){
        .in = {.control = &b->control, .response = &b->sram_response}, .out.selected = &b->biases};
    d->reg_cfg.ports = (Reg128Ports){
        .in = {.control = &b->control, .response = &b->sram_response}, .out.selected = &b->config};
    d->array.ports = (MacPorts){
        .in = {.control = &b->control, .weights = &b->weights, .inputs = &b->inputs, .biases = &b->biases}, .out.sums = &b->mac_sums};
    d->post.ports = (PostPorts){
        .in = {.control = &b->control, .sums = &b->mac_sums, .config = &b->config}, .out.result = &b->post_result};
}

bool npu_init_options(Npu *n, NpuOptions options)
{
    if (!n)
        return false;
    *n = (Npu){.cycle = 1};
    const LayerConfig config[NETWORK_LAYERS] = {
        {SRAM_L1_WEIGHTS, SRAM_L1_BIASES, SRAM_ACTIVATION_A, SRAM_ACTIVATION_B, SRAM_L1_CONFIG},
        {SRAM_L2_WEIGHTS, SRAM_L2_BIASES, SRAM_ACTIVATION_B, SRAM_ACTIVATION_A, SRAM_L2_CONFIG}};
    if (!controller_init(&n->controller, config, options))
        return false;
    DataPath *d = &n->data;
    sram_init(&d->sram);
    d->reg_w = (Reg128){.name = "Weight banks", .type = TYPE_WEIGHT};
    d->reg_i = (Reg128){.name = "Input banks", .type = TYPE_INPUT};
    d->reg_b = (Reg128){.name = "Bias banks", .type = TYPE_BIAS};
    d->reg_cfg = (Reg128){.name = "Config banks", .type = TYPE_CONFIG};
    wire_blocks(n);
    return true;
}

bool npu_init(Npu *n)
{
    return npu_init_options(n, (NpuOptions){NPU_BATCH, NPU_PREFETCH != 0, NPU_FUSE != 0});
}

bool npu_step(Npu *n)
{
    if (!n || n->error || n->controller.regs.done)
        return false;
    DataPath *d = &n->data;
    /* In evaluation order. All clock callbacks read the same frozen buses. */
    const Block blocks[] = {
        {&n->controller, &controller_ops},
        {&d->sram, &sram_ops},
        {&d->reg_w, &reg128_ops},
        {&d->reg_i, &reg128_ops},
        {&d->reg_b, &reg128_ops},
        {&d->reg_cfg, &reg128_ops},
        {&d->array, &mac_ops},
        {&d->post, &post_ops},
    };
    if (logging_ports_enabled())
        log_cycle_begin(n->cycle);
    if (!framework_step(&n->bus_context, blocks, sizeof blocks / sizeof blocks[0], n->cycle))
    {
        n->error = true;
        log_error(n->cycle, "NPU", "scheduled edge violated a datapath assertion");
        return false;
    }
    n->cycle++;
    return true;
}

NpuRunStatus npu_run(Npu *n, uint64_t limit)
{
    if (!n || n->error)
        return NPU_RUN_ERROR;
    for (uint64_t i = 0; i < limit && !n->controller.regs.done; i++)
    {
        if (!npu_step(n))
            return NPU_RUN_ERROR;
    }
    if (!n->controller.regs.done)
        return NPU_RUN_LIMIT;
    const DataPath *d = &n->data;
    unsigned long long cycles = (unsigned long long)(n->cycle - 1);
    log_flow(n->cycle - 1, "NPU", "COMPLETE", "cycles=%llu vectors=%u reads=%u writes=%u useful_mac_updates=%u compute_clocks=%u peak_active=%u/16 overlap_reads=%u hidden_forwards=%u",
             cycles, n->controller.options.batch, d->sram.regs.reads, d->sram.regs.writes, d->array.regs.updates,
             d->array.regs.compute_cycles, d->array.regs.peak_active, n->controller.regs.overlap_reads, d->reg_i.regs.forwards);
    log_flow(n->cycle - 1, "NPU", "UTILIZATION", "compute_only=%.2f%% whole_run=%.2f%%",
             100.0 * d->array.regs.updates / (16 * d->array.regs.compute_cycles), 100.0 * d->array.regs.updates / (16 * cycles));
    return NPU_RUN_COMPLETE;
}
