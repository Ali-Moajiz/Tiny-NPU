#include "fifo.h"
#include "logging.h"
void fifoBusses(void *context,uint64_t cycle)
{
    ResultFifo *f=context;Control c;Quantized q;
    log_block_begin("Result FIFO (depth 1)");
    if(bus_read(f->ports.control,&c,sizeof c)) {
        if(c.kind==EDGE_POST) {
            if(f->valid || !bus_read(f->ports.quantized,&q,sizeof q)) { f->error=true;log_error(cycle,"FIFO","invalid push or full FIFO");return; }
            f->word=pack_i8(q.row);f->valid=1;f->pushes++;
            Line128 write_data=line_broadcast(f->word); /* {q,q,q,q}: next layer needs I[r][c]=x[c] in every row */
            bus_write(f->ports.write_data,&write_data,sizeof write_data);
            log_flow(cycle,"FIFO","PUSH","word=0x%08X valid=1",f->word);
        } else if(c.kind==EDGE_STORE) {
            if(!f->valid) { f->error=true;log_error(cycle,"FIFO","pop while empty");return; }
            f->valid=0;f->pops++;log_flow(cycle,"FIFO","POP","packed SRAM store valid=0");
        }
    }
    bus_write(f->ports.valid,&f->valid,sizeof f->valid);
    log_port("CONTROL",f->ports.control,LOG_INPUT,LOG_BYTES);
    log_port("QUANTIZED",f->ports.quantized,LOG_INPUT,LOG_I8X4);
    log_port("SRAM_WRITE_DATA",f->ports.write_data,LOG_OUTPUT,LOG_U128);
    log_port("NEW_DATA",f->ports.valid,LOG_OUTPUT,LOG_U8);
}
bool fifo_init(ResultFifo *f,BusNetwork *n,const Bus *ctrl,const Bus *q)
{
    *f=(ResultFifo){0};f->ports.control=ctrl;f->ports.quantized=q;f->ports.write_data=&f->data_bus;f->ports.valid=&f->valid_bus;
    return bus_init(&f->data_bus,sizeof(Line128)) && bus_init(&f->valid_bus,sizeof(uint8_t)) && bus_register_link(n,&f->data_bus) && bus_register_link(n,&f->valid_bus) && bus_register_module(n,fifoBusses,f);
}
