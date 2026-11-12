#include "activation.h"
#include "logging.h"
#include <stdio.h>
static bool evaluate(void *instance,uint64_t cycle)
{
    Activation *a=instance;Control c=Control_read(a->ports.in.control);AccSignal out={0};
    if(c.post_en) {
        out=AccSignal_read(a->ports.in.sums);Line128 cfg=out.config;
        if(!out.valid||out.neurons!=c.neurons||out.batch!=c.batch||!cfg.lane[0]||cfg.lane[0]>INT32_MAX||cfg.lane[1]>62||cfg.lane[2]>1||cfg.lane[3]) {
            log_error(cycle,"Activation","POST requires matching live accumulators and valid CFG");return false;
        }
        if(cfg.lane[2]) for(unsigned r=0;r<c.neurons;r++) for(unsigned v=0;v<c.batch;v++)
            if(out.tile.value[r][v]<0) out.tile.value[r][v]=0;
    }
    AccSignal_drive(a->ports.out.activated,out);return true;
}
static void log_ports(void *instance,uint64_t cycle)
{
    Activation *a=instance;Control c=Control_read(a->ports.in.control);
    if(logging_ports_enabled()) {
        log_block_begin("Activation");
        printf("  post_en=%u in_valid=%u out_valid=%u\n",c.post_en,AccSignal_read(a->ports.in.sums).valid,AccSignal_read(a->ports.out.activated).valid);
        log_tile("in",AccSignal_read(a->ports.in.sums).tile);
        log_tile("out",AccSignal_read(a->ports.out.activated).tile);
        log_line("CFG",AccSignal_read(a->ports.in.sums).config);
    }
    if(c.post_en) log_flow(cycle,"Activation","POST","pc=0x%05X mode=%s",c.pc,AccSignal_read(a->ports.in.sums).config.lane[2]?"ReLU":"bypass");
}
const BlockOps activation_ops={.eval=evaluate,.clock=log_ports};
