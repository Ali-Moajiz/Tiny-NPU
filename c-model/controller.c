#include "controller.h"
#include "sram.h"
#include "logging.h"
#include <stdio.h>
static const char *type_name(TypeSel t) {
    static const char *names[]={"W(00)","I(01)","B(10)","CFG(11)"};
    return names[t];
}
static bool issue(Controller *c,Control *out,unsigned layer,TypeSel type,uint64_t cycle) {
    const LayerConfig *l=&c->layers[layer];
    uint32_t addr=type==TYPE_WEIGHT?l->weight_base:type==TYPE_INPUT?l->input_base:
                  type==TYPE_BIAS?l->bias_base:l->config_base;
    if(!sram_address_valid(addr) || !sram_address_matches_type(addr,type)) {
        c->error=true;log_error(cycle,"AGU","invalid address/type: 0x%03X %s",addr,type_name(type));return false;
    }
    out->read_enable=true;out->read_type=type;out->read_bank=layer%2;out->address=addr;
    log_flow(cycle,"AGU","READ_ISSUE","layer=%u dest_bank=%u type=%s line_addr=0x%03X%s",
             layer+1,out->read_bank,type_name(type),addr,out->kind==EDGE_COMPUTE?" background prefetch":"");
    return true;
}
static bool ready(const Controller *c) {
    const Bus *ports[]={c->ports.weights,c->ports.inputs,c->ports.biases,c->ports.config};
    for(unsigned i=0;i<4;i++) {
        RegBanks q;
        if(!bus_read(ports[i],&q,sizeof q) || !q.bank[c->layer%2].loaded) return false;
    }
    return true;
}
static void issue_parameter(Controller *c,Control *out,unsigned layer,uint64_t cycle) {
    static const TypeSel types[]={TYPE_WEIGHT,TYPE_BIAS,TYPE_CONFIG};
    if(c->issued[layer]<3 && issue(c,out,layer,types[c->issued[layer]],cycle)) c->issued[layer]++;
}
void controllerBusses(void *context,uint64_t cycle) {
    Controller *c=context;
    log_block_begin("Controller / prefetch AGU");
    if(c->done || c->error) return;
    Control out={.kind=EDGE_WAIT,.layer=c->layer,.bank=c->layer%2,
                 .batch=c->options.batch,.fuse=c->options.fuse};
    switch(c->state) {
    case CTRL_COLD: {
        static const TypeSel types[]={TYPE_WEIGHT,TYPE_INPUT,TYPE_BIAS,TYPE_CONFIG};
        out.kind=EDGE_LOAD;
        if(!issue(c,&out,0,types[c->cold_read],cycle)) return;
        if(++c->cold_read==4) {c->issued[0]=3;c->state=CTRL_READY;}
        break;
    }
    case CTRL_READY:
        /* Observe actual registered loaded flags, not just elapsed ticks. */
        if(c->issued[c->layer]<3) issue_parameter(c,&out,c->layer,cycle);
        else if(ready(c)) {
            out.kind=EDGE_COMPUTE;out.wave=1;c->wave=2;c->state=CTRL_COMPUTE;
            log_flow(cycle,"AGU","BANK_SELECT","layer=%u active_bank=%u all W/I/B/CFG ready",c->layer+1,out.bank);
        } else c->wait_cycles++;
        break;
    case CTRL_COMPUTE:
        out.kind=EDGE_COMPUTE;out.wave=c->wave++;
        if(out.wave==COMPUTE_WAVES) c->state=CTRL_ACTIVATE;
        break;
    case CTRL_ACTIVATE: out.kind=EDGE_ACTIVATE;c->state=CTRL_SCALE;break;
    case CTRL_SCALE: out.kind=EDGE_SCALE;c->state=CTRL_HANDOFF;break;
    case CTRL_HANDOFF:
        out.kind=EDGE_HANDOFF;
        if(c->layer+1<NETWORK_LAYERS && c->options.fuse) {
            c->layer++;c->state=CTRL_READY;
        } else c->state=CTRL_STORE;
        break;
    case CTRL_STORE: {
        uint8_t valid=0;
        if(!bus_read(c->ports.fifo_valid,&valid,sizeof valid) || !valid) {
            c->error=true;log_error(cycle,"AGU","STORE without registered FIFO valid");return;
        }
        out.kind=EDGE_STORE;out.address=c->layers[c->layer].output_base;
        out.byte_enable=batch_byte_enable(c->options.batch);
        if(!sram_address_valid(out.address) || !sram_address_matches_type(out.address,TYPE_INPUT)) {
            c->error=true;log_error(cycle,"AGU","invalid result address");return;
        }
        if(c->layer+1==NETWORK_LAYERS) c->done=true;
        else {c->layer++;c->state=CTRL_RELOAD_INPUT;}
        break;
    }
    case CTRL_RELOAD_INPUT:
        out.kind=EDGE_LOAD;
        if(!issue(c,&out,c->layer,TYPE_INPUT,cycle)) return;
        c->state=CTRL_READY;break;
    }
    /* AGU prefetch is independent of computation. The single SRAM port still
     * accepts at most ONE read OR write per clock. Destination bank tags ensure
     * shadow captures cannot overwrite parameters used by the active layer. */
    if(out.kind==EDGE_COMPUTE && c->options.prefetch && out.layer+1<NETWORK_LAYERS && c->issued[out.layer+1]<3) {
        issue_parameter(c,&out,out.layer+1,cycle);
        if(out.read_enable) c->overlap_reads++;
    }
    if(c->error) return;
    bus_write(c->ports.control,&out,sizeof out);
    static const char *phases[]={"WAIT","LOAD","COMPUTE","ACTIVATE","SCALE","HANDOFF","STORE"};
    log_flow(cycle,"AGU","EDGE","layer=%u bank=%u phase=%s k=%u read=%u batch=%u",
             out.layer+1,out.bank,phases[out.kind],out.wave,out.read_enable,out.batch);
    if(logging_ports_enabled()) printf("  PHASE=%s LAYER=%u ACTIVE_BANK=%u K=%u BATCH=%u RD=%u TYPE=%u DEST_BANK=%u ADDR=0x%03X BE=0x%04X FUSE=%u\n",
        phases[out.kind],out.layer+1,out.bank,out.wave,out.batch,out.read_enable,(unsigned)out.read_type,out.read_bank,out.address,out.byte_enable,out.fuse);
    log_port("CONTROL",c->ports.control,LOG_OUTPUT,LOG_BYTES);
    log_port("FIFO_VALID",c->ports.fifo_valid,LOG_INPUT,LOG_U8);
    log_port("W_BANKS",c->ports.weights,LOG_INPUT,LOG_BYTES);
    log_port("I_BANKS",c->ports.inputs,LOG_INPUT,LOG_BYTES);
    log_port("B_BANKS",c->ports.biases,LOG_INPUT,LOG_BYTES);
    log_port("CFG_BANKS",c->ports.config,LOG_INPUT,LOG_BYTES);
}
bool controller_init(Controller *c,BusNetwork *n,const LayerConfig cfg[NETWORK_LAYERS],
                     NpuOptions options,const Bus *valid,const Bus *w,const Bus *x,const Bus *b,const Bus *q) {
    if(!c || !n || !cfg || !valid || !w || !x || !b || !q || options.batch<1 || options.batch>4) return false;
    *c=(Controller){0};c->options=options;c->ports.control=&c->control_bus;
    c->ports.fifo_valid=valid;c->ports.weights=w;c->ports.inputs=x;c->ports.biases=b;c->ports.config=q;
    for(unsigned i=0;i<NETWORK_LAYERS;i++) c->layers[i]=cfg[i];
    return bus_init_wire(&c->control_bus,sizeof(Control)) && bus_register_link(n,&c->control_bus) && bus_register_module(n,controllerBusses,c);
}
