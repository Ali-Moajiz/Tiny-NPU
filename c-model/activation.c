#include "activation.h"
#include "logging.h"
void activationBusses(void *context,uint64_t cycle) {
    Activation *a=context;Control c;TileSums sums;RegBanks cfg;
    log_block_begin("Activation: 4 outputs x up to 4 vectors (registered)");
    if(bus_read(a->ports.control,&c,sizeof c) && c.kind==EDGE_ACTIVATE) {
        if(c.bank>=2 || !bus_read(a->ports.row_sums,&sums,sizeof sums) ||
           !bus_read(a->ports.config,&cfg,sizeof cfg) || !cfg.bank[c.bank].loaded) {
            a->error=true;log_error(cycle,"Activation","missing final accumulators/config");return;
        }
        bool relu=cfg.bank[c.bank].line.lane[2]!=0;
        for(unsigned r=0;r<4;r++) for(unsigned v=0;v<4;v++)
            if(v>=c.batch || (relu && sums.value[r][v]<0)) sums.value[r][v]=0;
        bus_write(a->ports.activated,&sums,sizeof sums);
        for(unsigned v=0;v<c.batch;v++) log_flow(cycle,"Activation","REGISTER","layer=%u vector=%u relu=%u values=[%d,%d,%d,%d] valid_cycle=%llu",
            c.layer+1,v+1,relu,sums.value[0][v],sums.value[1][v],sums.value[2][v],sums.value[3][v],(unsigned long long)(cycle+1));
    }
    log_port("CONTROL",a->ports.control,LOG_INPUT,LOG_BYTES);
    log_port("FINAL_ACC_TILE",a->ports.row_sums,LOG_INPUT,LOG_I32X16);
    log_port("CFG_BANKS",a->ports.config,LOG_INPUT,LOG_BYTES);
    log_port("ACTIVATED_TILE",a->ports.activated,LOG_OUTPUT,LOG_I32X16);
}
bool activation_init(Activation *a,BusNetwork *n,const Bus *ctrl,const Bus *sums,const Bus *config) {
    if(!a || !ctrl || !sums || !config) return false;
    *a=(Activation){0};a->ports.control=ctrl;a->ports.row_sums=sums;a->ports.config=config;a->ports.activated=&a->output_bus;
    return bus_init(&a->output_bus,sizeof(TileSums)) && bus_register_link(n,&a->output_bus) && bus_register_module(n,activationBusses,a);
}
