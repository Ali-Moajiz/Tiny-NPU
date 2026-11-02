#include "fifo.h"
#include "logging.h"
void fifoBusses(void *context,uint64_t cycle) {
    ResultFifo *f=context;Control c;
    log_block_begin("Result FIFO / hold (one 128-bit tile)");
    if(bus_read(f->ports.control,&c,sizeof c)) {
        if(c.kind==EDGE_HANDOFF && (!c.fuse || c.layer+1==NETWORK_LAYERS)) {
            if(f->valid || !bus_read(f->ports.quantized,&f->tile,sizeof f->tile)) {
                f->error=true;log_error(cycle,"FIFO","push while full or missing quantized tile");return;
            }
            f->valid=1;f->pushes++;
            log_flow(cycle,"FIFO","PUSH","layer=%u vectors=%u valid_cycle=%llu",c.layer+1,c.batch,(unsigned long long)(cycle+1));
        } else if(c.kind==EDGE_STORE) {
            if(!f->valid) {f->error=true;log_error(cycle,"FIFO","pop while empty");return;}
            f->valid=0;f->pops++;log_flow(cycle,"FIFO","POP","tile consumed by SRAM store");
        }
    }
    bus_write(f->ports.write_data,&f->tile,sizeof f->tile);
    bus_write(f->ports.valid,&f->valid,sizeof f->valid);
    log_port("CONTROL",f->ports.control,LOG_INPUT,LOG_BYTES);
    log_port("QUANTIZED_TILE",f->ports.quantized,LOG_INPUT,LOG_U128);
    log_port("SRAM_WRITE_DATA",f->ports.write_data,LOG_OUTPUT,LOG_U128);
    log_port("FIFO_VALID",f->ports.valid,LOG_OUTPUT,LOG_U8);
}
bool fifo_init(ResultFifo *f,BusNetwork *n,const Bus *ctrl,const Bus *q) {
    if(!f || !ctrl || !q) return false;
    *f=(ResultFifo){0};f->ports.control=ctrl;f->ports.quantized=q;f->ports.write_data=&f->data_bus;f->ports.valid=&f->valid_bus;
    return bus_init(&f->data_bus,sizeof(Line128)) && bus_init(&f->valid_bus,sizeof(uint8_t)) && bus_register_link(n,&f->data_bus) && bus_register_link(n,&f->valid_bus) && bus_register_module(n,fifoBusses,f);
}
