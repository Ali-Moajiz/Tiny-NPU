#include "mac.h"
#include "logging.h"
#include <stdio.h>
#include <inttypes.h>
static void compute(MacArray *a,const Control *c,uint64_t cycle) {
    RegBanks w,x,b;
    if(c->bank>=2 || c->batch<1 || c->batch>4 || c->wave<1 || c->wave>4 ||
       !bus_read(a->ports.weights,&w,sizeof w) || !bus_read(a->ports.inputs,&x,sizeof x) ||
       !bus_read(a->ports.biases,&b,sizeof b) || !w.bank[c->bank].loaded || !x.bank[c->bank].loaded || !b.bank[c->bank].loaded ||
       c->wave!=a->next_wave) {
        a->error=true;log_error(cycle,"MAC_array","invalid compute order or unloaded operand bank");return;
    }
    TileSums next={0};
    unsigned k=c->wave-1;
    /* Every result reads its OWN pre-edge accumulator, not an updated neighbour.
     * The loop describes 16 parallel PEs, not 16 modeled clock cycles. */
    for(unsigned r=0;r<4;r++) for(unsigned v=0;v<c->batch;v++) {
        int8_t weight=unpack_i8(w.bank[c->bank].line.lane[r],k);
        int8_t input=unpack_i8(x.bank[c->bank].line.lane[v],k);
        if(k && !a->cell[r][v].valid) {
            a->error=true;log_error(cycle,"MAC_array","missing prior accumulator");return;
        }
        int32_t incoming=k?a->cell[r][v].psum:unpack_i32(b.bank[c->bank].line.lane[r]);
        int16_t product=(int16_t)((int)weight*input);
        int64_t sum=(int64_t)incoming+product;
        if(sum<INT32_MIN || sum>INT32_MAX) {
            a->error=true;log_error(cycle,"MAC_array","INT32 overflow neuron=%u vector=%u",r+1,v+1);return;
        }
        next.value[r][v]=(int32_t)sum;
        log_flow(cycle,"MAC_array","MAC_UPDATE","L%u C%u PE(neuron=%u,vector=%u) x=%d w=%d acc_in=%" PRId32 "%s product=%d acc_out=%" PRId32 " done=%u",
                 c->layer+1,c->wave,r+1,v+1,input,weight,incoming,k?"":"(bias mux)",product,(int32_t)sum,k==3);
    }
    /* Commit all local PE registers after evaluating every next value. */
    for(unsigned r=0;r<4;r++) for(unsigned v=0;v<4;v++) {
        a->cell[r][v].psum=next.value[r][v];a->cell[r][v].valid=v<c->batch;
    }
    a->sums=next;a->compute_cycles++;a->updates+=4*c->batch;
    if(a->peak_active<4*c->batch) a->peak_active=4*c->batch;
    a->next_wave=k==3?1:c->wave+1;
    if(k==3) bus_write(a->ports.row_sums,&a->sums,sizeof a->sums);
    log_flow(cycle,"MAC_array","UTILIZATION","active=%u/16 useful_MACs_this_clock=%u k=%u/4",4*c->batch,4*c->batch,c->wave);
}
void macArrayBusses(void *context,uint64_t cycle) {
    MacArray *a=context;Control c;
    log_block_begin("4x4 outer-product MAC array / local INT32 accumulators");
    if(bus_read(a->ports.control,&c,sizeof c) && c.kind==EDGE_COMPUTE) compute(a,&c,cycle);
    log_port("CONTROL",a->ports.control,LOG_INPUT,LOG_BYTES);
    log_port("W_BANKS",a->ports.weights,LOG_INPUT,LOG_BYTES);
    log_port("I_BANKS",a->ports.inputs,LOG_INPUT,LOG_BYTES);
    log_port("B_BANKS",a->ports.biases,LOG_INPUT,LOG_BYTES);
    log_port("FINAL_ACC_TILE",a->ports.row_sums,LOG_OUTPUT,LOG_I32X16);
    if(logging_ports_enabled()) for(unsigned r=0;r<4;r++) for(unsigned v=0;v<4;v++)
        printf("  PE(neuron=%u,vector=%u) acc=%" PRId32 " valid=%u\n",r+1,v+1,a->cell[r][v].psum,a->cell[r][v].valid);
}
bool mac_array_init(MacArray *a,BusNetwork *n,const Bus *control,const Bus *w,const Bus *x,const Bus *b) {
    if(!a || !control || !w || !x || !b) return false;
    *a=(MacArray){0};a->next_wave=1;a->ports.control=control;a->ports.weights=w;a->ports.inputs=x;a->ports.biases=b;a->ports.row_sums=&a->sums_bus;
    return bus_init(&a->sums_bus,sizeof(TileSums)) && bus_register_link(n,&a->sums_bus) && bus_register_module(n,macArrayBusses,a);
}
