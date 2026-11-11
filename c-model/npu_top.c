/* Top-level netlist. This is the only place that connects module ports. */
#include "npu.h"
#include "logging.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
void npu_init(Npu *n)
{
    memset(n,0,sizeof *n);controller_init(&n->controller);
    n->stream.regs.output_enabled=true;
    Control_connect(&n->wires.control,&n->bus_context,"controller.control",&n->controller);
    TileSignal_connect(&n->wires.memory,&n->bus_context,"sram.response",&n->sram);
    TileSignal_connect(&n->wires.input,&n->bus_context,"stream.input",&n->stream);
    StreamStatus_connect(&n->wires.stream_status,&n->bus_context,"stream.status",&n->stream);
    AccSignal_connect(&n->wires.sums,&n->bus_context,"mac.accumulators",&n->mac);
    AccSignal_connect(&n->wires.activated,&n->bus_context,"activation.result",&n->activation);
    ResultSignal_connect(&n->wires.result,&n->bus_context,"scale.result",&n->scale);
    n->controller.ports.in.memory=&n->wires.memory;n->controller.ports.in.stream=&n->wires.stream_status;
    n->controller.ports.out.control=&n->wires.control;
    n->sram.ports.in.control=&n->wires.control;n->sram.ports.in.input=&n->wires.input;
    n->sram.ports.in.result=&n->wires.result;n->sram.ports.out.response=&n->wires.memory;
    n->stream.ports.in.control=&n->wires.control;n->stream.ports.in.response=&n->wires.memory;
    n->stream.ports.out.input=&n->wires.input;n->stream.ports.out.status=&n->wires.stream_status;
    n->mac.ports.in.control=&n->wires.control;n->mac.ports.in.response=&n->wires.memory;
    n->mac.ports.in.forward=&n->wires.result;n->mac.ports.out.sums=&n->wires.sums;
    n->activation.ports.in.control=&n->wires.control;n->activation.ports.in.sums=&n->wires.sums;
    n->activation.ports.out.activated=&n->wires.activated;
    n->scale.ports.in.control=&n->wires.control;n->scale.ports.in.activated=&n->wires.activated;
    n->scale.ports.out.result=&n->wires.result;
}
bool npu_step(Npu *n)
{
    if(n->error||n->controller.regs.done) return false;
    /* EVAL order follows combinational dependencies; every register update
     * happens only after ALL blocks have passed CHECK. No same-edge leakage. */
    const Block blocks[]={
        {&n->controller,&controller_ops},{&n->sram,&sram_ops},{&n->stream,&stream_ops},
        {&n->mac,&mac_ops},{&n->activation,&activation_ops},{&n->scale,&scale_clamp_ops}
    };
    n->cycle++;log_cycle_begin(n->cycle);
    if(!framework_step(&n->bus_context,blocks,sizeof blocks/sizeof blocks[0],n->cycle)) n->error=true;
    return !n->error;
}
bool npu_run(Npu *n,uint64_t max_cycles)
{
    while(!n->controller.regs.done&&!n->error&&n->cycle<max_cycles) (void)npu_step(n);
    if(!n->error&&!n->controller.regs.done) {
        log_error(n->cycle,"NPU","cycle limit: pc=0x%04X (missing input, back-pressure, loop or missing HALT)",n->controller.regs.pc);n->error=true;
    }
    return !n->error;
}
void npu_print_metrics(const Npu *n)
{
    double compute=n->mac.regs.compute_cycles?100.0*(double)n->mac.regs.useful_macs/(16.0*(double)n->mac.regs.compute_cycles):0;
    double whole=n->cycle?100.0*(double)n->mac.regs.useful_macs/(16.0*(double)n->cycle):0;
    printf("METRICS cycles=%" PRIu64 " instructions=%" PRIu64 " compute_cycles=%" PRIu64 " useful_macs=%" PRIu64 " bus_stall_cycles=%" PRIu64 " sram_reads=%" PRIu64 " sram_writes=%" PRIu64 " compute_util=%.2f%% whole_util=%.2f%%\n",
           n->cycle,n->controller.regs.instructions,n->mac.regs.compute_cycles,n->mac.regs.useful_macs,n->controller.regs.bus_stall_cycles,n->sram.regs.reads,n->sram.regs.writes,compute,whole);
}
