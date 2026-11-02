#include "npu.h"
#include "logging.h"
bool npu_init_options(Npu *n,NpuOptions options) {
    if(!n || options.batch<1 || options.batch>4) return false;
    *n=(Npu){0};bus_network_init(&n->network);bus_network_set_logging(&n->network,true);n->network.cycle=1;
    const LayerConfig config[NETWORK_LAYERS]={
        {SRAM_L1_WEIGHTS,SRAM_L1_BIASES,SRAM_ACTIVATION_A,SRAM_ACTIVATION_B,SRAM_L1_CONFIG},
        {SRAM_L2_WEIGHTS,SRAM_L2_BIASES,SRAM_ACTIVATION_B,SRAM_ACTIVATION_A,SRAM_L2_CONFIG}};
    const Bus *ctrl=&n->controller.control_bus;
    /* Preserve the generic callback architecture. Registered links expose the
     * pre-edge snapshot even when their producer ran earlier this traversal.
     * Only control/demux outputs are wires. No generic framework changes. */
    if(!controller_init(&n->controller,&n->network,config,options,&n->fifo.valid_bus,&n->reg_w.out_bus,&n->reg_i.out_bus,&n->reg_b.out_bus,&n->reg_cfg.out_bus) ||
       !sram_init(&n->sram,&n->network,ctrl,&n->fifo.data_bus) ||
       !demux_init(&n->demux,&n->network,ctrl,&n->sram.response_bus) ||
       !reg128_init(&n->reg_w,&n->network,"Weight banks",TYPE_WEIGHT,ctrl,&n->demux.out[TYPE_WEIGHT],NULL) ||
       !reg128_init(&n->reg_i,&n->network,"Input banks",TYPE_INPUT,ctrl,&n->demux.out[TYPE_INPUT],&n->scale.output_bus) ||
       !reg128_init(&n->reg_b,&n->network,"Bias banks",TYPE_BIAS,ctrl,&n->demux.out[TYPE_BIAS],NULL) ||
       !reg128_init(&n->reg_cfg,&n->network,"Config banks",TYPE_CONFIG,ctrl,&n->demux.out[TYPE_CONFIG],NULL) ||
       !mac_array_init(&n->array,&n->network,ctrl,&n->reg_w.out_bus,&n->reg_i.out_bus,&n->reg_b.out_bus) ||
       !activation_init(&n->activation,&n->network,ctrl,&n->array.sums_bus,&n->reg_cfg.out_bus) ||
       !scale_clamp_init(&n->scale,&n->network,ctrl,&n->activation.output_bus,&n->reg_cfg.out_bus) ||
       !fifo_init(&n->fifo,&n->network,ctrl,&n->scale.output_bus)) return false;
    log_flow(0,"NPU","INITIALIZED","network=4->4->4 outer_product=4x4 batch=%u parameter_banks=2 prefetch=%u fuse=%u",options.batch,options.prefetch,options.fuse);
    return true;
}
bool npu_init(Npu *n) {
    return npu_init_options(n,(NpuOptions){NPU_BATCH,NPU_PREFETCH!=0,NPU_FUSE!=0});
}
NpuRunStatus npu_run(Npu *n,uint64_t limit) {
    if(!n || n->error) return NPU_RUN_ERROR;
    for(uint64_t i=0;i<limit && !n->controller.done;i++) {
        bool ok=bus_run_cycle(&n->network);
        if(!ok || n->controller.error || n->sram.error || n->demux.error || n->reg_w.error || n->reg_i.error || n->reg_b.error ||
           n->reg_cfg.error || n->array.error || n->activation.error || n->scale.error || n->fifo.error) {
            n->error=true;log_error(n->network.cycle-1,"NPU","stopping after block/bus fault");return NPU_RUN_ERROR;
        }
    }
    if(!n->controller.done) return NPU_RUN_LIMIT;
    unsigned long long cycles=(unsigned long long)(n->network.cycle-1);
    log_flow(n->network.cycle-1,"NPU","COMPLETE","cycles=%llu vectors=%u reads=%u writes=%u useful_mac_updates=%u compute_clocks=%u peak_active=%u/16 overlap_reads=%u hidden_forwards=%u",
             cycles,n->controller.options.batch,n->sram.reads,n->sram.writes,n->array.updates,n->array.compute_cycles,n->array.peak_active,n->controller.overlap_reads,n->reg_i.forwards);
    log_flow(n->network.cycle-1,"NPU","UTILIZATION","compute_only=%.2f%% whole_run=%.2f%% (includes loads/post/store)",
             100.0*n->array.updates/(16*n->array.compute_cycles),100.0*n->array.updates/(16*cycles));
    return NPU_RUN_COMPLETE;
}
