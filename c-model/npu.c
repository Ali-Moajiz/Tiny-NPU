#include "npu.h"
#include "logging.h"
bool npu_init(Npu *n)
{
    if (!n)
        return false;
    *n = (Npu){0};
    bus_network_init(&n->network);
    bus_network_set_logging(&n->network, true);
    n->network.cycle = 1; /* Cycle 1 is E1, after host start/config acceptance. */
    const LayerConfig config[NETWORK_LAYERS] = {
        {SRAM_L1_WEIGHTS, SRAM_L1_BIASES, SRAM_ACTIVATION_A, SRAM_ACTIVATION_B, SRAM_L1_CONFIG},
        {SRAM_L2_WEIGHTS, SRAM_L2_BIASES, SRAM_ACTIVATION_B, SRAM_ACTIVATION_A, SRAM_L2_CONFIG}};
    /* Modules run in registration order: controller, SRAM, demux, registers, array,
     * activation, scale, FIFO. SRAM Q is a registered link, so the demux (which runs
     * after SRAM) still sees the PREVIOUS clock's Q. Demux outputs are wires and the
     * registers run after the demux, so a read issued at clock N lands at clock N+1. */
    const Bus *ctrl = &n->controller.control_bus;
    if (!controller_init(&n->controller, &n->network, config, &n->fifo.valid_bus) ||
        !sram_init(&n->sram, &n->network, ctrl, &n->fifo.data_bus) ||
        !demux_init(&n->demux, &n->network, ctrl, &n->sram.response_bus) ||
        !reg128_init(&n->reg_w, &n->network, "Weight reg", TYPE_WEIGHT, ctrl, &n->demux.out[TYPE_WEIGHT]) ||
        !reg128_init(&n->reg_i, &n->network, "Input reg", TYPE_INPUT, ctrl, &n->demux.out[TYPE_INPUT]) ||
        !reg128_init(&n->reg_b, &n->network, "Bias reg", TYPE_BIAS, ctrl, &n->demux.out[TYPE_BIAS]) ||
        !reg128_init(&n->reg_cfg, &n->network, "Config reg", TYPE_CONFIG, ctrl, &n->demux.out[TYPE_CONFIG]) ||
        !mac_array_init(&n->array, &n->network, ctrl, &n->reg_w.out_bus, &n->reg_i.out_bus, &n->reg_b.out_bus) ||
        !activation_init(&n->activation, &n->network, ctrl, &n->array.sums_bus, &n->reg_cfg.out_bus) ||
        !scale_clamp_init(&n->scale, &n->network, ctrl, &n->activation.output_bus, &n->reg_cfg.out_bus) ||
        !fifo_init(&n->fifo, &n->network, ctrl, &n->scale.output_bus))
        return false;
    log_flow(0, "NPU", "INITIALIZED", "network=4->4->4 array=4x4 SRAM=1024x128 SRAM->demux->W/I/B/CFG regs->array layers=2");
    return true;
}
NpuRunStatus npu_run(Npu *n, uint64_t limit)
{
    if (!n)
        return NPU_RUN_ERROR;
    for (uint64_t i = 0; i < limit && !n->controller.done; i++)
    {
        bool ok = bus_run_cycle(&n->network);
        if (!ok || n->controller.error || n->sram.error || n->demux.error || n->reg_w.error || n->reg_i.error || n->reg_b.error ||
            n->reg_cfg.error || n->array.error || n->activation.error || n->scale.error || n->fifo.error)
        {
            log_error(n->network.cycle - 1, "NPU", "stopping after block/bus fault");
            return NPU_RUN_ERROR;
        }
    }
    if (!n->controller.done)
    {
        log_flow(n->network.cycle, "NPU", "RUN_LIMIT", "not complete");
        return NPU_RUN_LIMIT;
    }
    log_flow(n->network.cycle - 1, "NPU", "COMPLETE", "cycles=%llu reads=%u writes=%u mac_updates=%u output_line_addr=0x%03X word=0x%08X",
             (unsigned long long)(n->network.cycle - 1), n->sram.reads, n->sram.writes, n->array.updates, SRAM_ACTIVATION_A, n->sram.lines[SRAM_ACTIVATION_A].lane[0]);
    return NPU_RUN_COMPLETE;
}
