#include "scale_clamp.h"
#include "logging.h"
/* INT32 times positive INT32 fits INT64. Quotient/remainder implements
 * nearest-even rounding without signed shifts or rounding-bias overflow. */
int8_t scale_clamp_value(int32_t value,int32_t multiplier,unsigned shift) {
    int64_t product=(int64_t)value*multiplier;
    bool negative=product<0;
    uint64_t magnitude=(uint64_t)(negative?-product:product),rounded=magnitude;
    if(shift) {
        uint64_t divisor=UINT64_C(1)<<shift;
        uint64_t quotient=magnitude/divisor,remainder=magnitude%divisor,half=divisor/2;
        if(remainder>half || (remainder==half && (quotient&1))) quotient++;
        rounded=quotient;
    }
    if(negative) return rounded>=128?INT8_MIN:(int8_t)-(int64_t)rounded;
    return rounded>127?INT8_MAX:(int8_t)rounded;
}
void scaleClampBusses(void *context,uint64_t cycle) {
    ScaleClamp *s=context;Control c;TileSums sums;RegBanks cfg;
    log_block_begin("Requantize/clamp: registered INT8 tile");
    if(bus_read(s->ports.control,&c,sizeof c) && c.kind==EDGE_SCALE) {
        if(c.bank>=2 || !bus_read(s->ports.config,&cfg,sizeof cfg) || !cfg.bank[c.bank].loaded ||
           !bus_read(s->ports.activated,&sums,sizeof sums)) {
            s->error=true;log_error(cycle,"Scale","missing registered activation/config");return;
        }
        int32_t m=unpack_i32(cfg.bank[c.bank].line.lane[0]);unsigned shift=cfg.bank[c.bank].line.lane[1];
        if(m<=0 || shift>62) {s->error=true;log_error(cycle,"Scale","invalid M/S");return;}
        Line128 out={{0}};
        for(unsigned v=0;v<c.batch;v++) {
            int8_t q[4];
            for(unsigned r=0;r<4;r++) q[r]=scale_clamp_value(sums.value[r][v],m,shift);
            out.lane[v]=pack_i8(q);
            log_flow(cycle,"Scale","QUANTIZE","layer=%u vector=%u M=%d S=%u rounding=nearest_ties_even q=[%d,%d,%d,%d] valid_cycle=%llu",
                     c.layer+1,v+1,m,shift,q[0],q[1],q[2],q[3],(unsigned long long)(cycle+1));
        }
        bus_write(s->ports.quantized,&out,sizeof out);
    }
    log_port("CONTROL",s->ports.control,LOG_INPUT,LOG_BYTES);
    log_port("ACTIVATED_TILE",s->ports.activated,LOG_INPUT,LOG_I32X16);
    log_port("CFG_BANKS",s->ports.config,LOG_INPUT,LOG_BYTES);
    log_port("QUANTIZED_TILE",s->ports.quantized,LOG_OUTPUT,LOG_U128);
}
bool scale_clamp_init(ScaleClamp *s,BusNetwork *n,const Bus *ctrl,const Bus *act,const Bus *config) {
    if(!s || !ctrl || !act || !config) return false;
    *s=(ScaleClamp){0};s->ports.control=ctrl;s->ports.activated=act;s->ports.config=config;s->ports.quantized=&s->output_bus;
    return bus_init(&s->output_bus,sizeof(Line128)) && bus_register_link(n,&s->output_bus) && bus_register_module(n,scaleClampBusses,s);
}
