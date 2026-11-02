#include "reg128.h"
#include "logging.h"
#include <stdio.h>
static bool config_valid(const Line128 *q) {
    return q->lane[0] && q->lane[0]<=INT32_MAX && q->lane[1]<=62 && q->lane[2]<=1 && !q->lane[3];
}
void reg128Busses(void *context,uint64_t cycle) {
    Reg128 *r=context;Control c;TaggedLine d={0};
    log_block_begin(r->name);
    bool have_control=bus_read(r->ports.control,&c,sizeof c);
    bool capture=bus_read(r->ports.in,&d,sizeof d);
    bool forward=have_control && r->type==TYPE_INPUT && c.kind==EDGE_HANDOFF && c.fuse && c.layer+1<NETWORK_LAYERS;
    if(forward) {
        if(capture || !bus_read(r->ports.forwarded,&d.line,sizeof d.line)) {
            r->error=true;log_error(cycle,r->name,"forwarding mux collision or missing registered quantized tile");return;
        }
        d.bank=(c.layer+1)%2;capture=true;r->forwards++;
    }
    if(capture) {
        if(d.bank>=PARAM_BANKS || (r->type==TYPE_CONFIG && !config_valid(&d.line))) {
            r->error=true;log_error(cycle,r->name,"invalid bank or M/S/ReLU configuration");return;
        }
        if(have_control && c.kind>=EDGE_COMPUTE && c.kind<=EDGE_HANDOFF && d.bank==c.bank) {
            r->error=true;log_error(cycle,r->name,"attempt to overwrite active bank");return;
        }
        r->value.bank[d.bank].line=d.line;r->value.bank[d.bank].loaded=1;r->captures++;
        log_flow(cycle,r->name,forward?"FORWARD_HIDDEN":"CAPTURE","bank=%u line=%08X_%08X_%08X_%08X%s",
                 d.bank,d.line.lane[3],d.line.lane[2],d.line.lane[1],d.line.lane[0],forward?" SRAM bypass (already INT8 requantized)":"");
    }
    bus_write(r->ports.out,&r->value,sizeof r->value);
    log_port("CONTROL",r->ports.control,LOG_INPUT,LOG_BYTES);
    log_port("SRAM_D + BANK",r->ports.in,LOG_INPUT,LOG_BYTES);
    if(r->type==TYPE_INPUT) log_port("FORWARDED_INT8_TILE",r->ports.forwarded,LOG_INPUT,LOG_U128);
    log_port("BANKED_Q",r->ports.out,LOG_OUTPUT,LOG_BYTES);
    if(logging_ports_enabled()) for(unsigned bank=0;bank<2;bank++) {
        const RegValue *q=&r->value.bank[bank];
        printf("  BANK%u D/next=%08X_%08X_%08X_%08X loaded=%u\n",bank,q->line.lane[3],q->line.lane[2],q->line.lane[1],q->line.lane[0],q->loaded);
    }
}
bool reg128_init(Reg128 *r,BusNetwork *n,const char *name,TypeSel type,const Bus *control,const Bus *in,const Bus *forwarded) {
    if(!r || !name || !control || !in || (type==TYPE_INPUT && !forwarded)) return false;
    *r=(Reg128){0};r->name=name;r->type=type;r->ports.control=control;r->ports.in=in;r->ports.forwarded=forwarded;r->ports.out=&r->out_bus;
    return bus_init(&r->out_bus,sizeof(RegBanks)) && bus_register_link(n,&r->out_bus) && bus_register_module(n,reg128Busses,r);
}
