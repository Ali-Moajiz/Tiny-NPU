#include "mac.h"
#include "logging.h"
#include <stdio.h>
static bool evaluate(void *instance,uint64_t cycle)
{
    (void)cycle;Mac *m=instance;Control c=Control_read(m->ports.in.control);
    AccSignal sums={0};sums.tile=m->regs.accumulators;
    sums.valid=m->regs.valid&&m->regs.complete;
    sums.neurons=m->regs.neurons;sums.batch=m->regs.batch;
    if(c.slot<OPERAND_SLOTS) {
        sums.config=m->regs.slot[c.slot].operand[OPERAND_CFG];
        if(!m->regs.slot[c.slot].loaded[OPERAND_CFG]) sums.valid=false;
    } else sums.valid=false;
    AccSignal_drive(m->ports.out.sums,sums);
    return true;
}
static bool check(void *instance,uint64_t cycle)
{
    Mac *m=instance;Control c=Control_read(m->ports.in.control);
    if(c.latch_en) {
        if(c.latch_slot>=OPERAND_SLOTS||c.latch_field>OPERAND_CFG||!TileSignal_read(m->ports.in.response).valid||c.compute_en||c.forward_en) goto fault;
        if(c.latch_field==OPERAND_CFG) {
            Line128 cfg=TileSignal_read(m->ports.in.response).tile;
            if(!cfg.lane[0]||cfg.lane[0]>INT32_MAX||cfg.lane[1]>62||cfg.lane[2]>1||cfg.lane[3]) goto fault;
        }
    }
    if(c.forward_en) {
        ResultSignal result=ResultSignal_read(m->ports.in.forward);
        if(c.forward_slot>=OPERAND_SLOTS||!result.valid||result.neurons!=c.neurons||result.batch!=c.batch||c.compute_en) goto fault;
    }
    if(!c.compute_en) return true;
    if(c.slot>=OPERAND_SLOTS||!c.neurons||c.neurons>4||!c.batch||c.batch>4||!c.k_count||c.k_count>4||c.k>=c.k_count||c.post_en) goto fault;
    OperandSlot *s=&m->regs.slot[c.slot];
    if(!s->loaded[OPERAND_X]||!s->loaded[OPERAND_W]||(!c.accumulate&&!s->loaded[OPERAND_B])) goto fault;
    if(c.k==0) {
        if(c.accumulate&&(!m->regs.valid||!m->regs.complete||m->regs.neurons!=c.neurons||m->regs.batch!=c.batch)) goto fault;
    } else if(!m->regs.valid||m->regs.complete||m->regs.next_k!=c.k||m->regs.neurons!=c.neurons||m->regs.batch!=c.batch) goto fault;
    m->products_added=(TileSums){0};
    for(unsigned r=0;r<c.neurons;r++) for(unsigned v=0;v<c.batch;v++) {
        int32_t prior=(c.k==0&&!c.accumulate)?unpack_i32(s->operand[OPERAND_B].lane[r]):m->regs.accumulators.value[r][v];
        int x=line_byte(s->operand[OPERAND_X],v*4+c.k),w=line_byte(s->operand[OPERAND_W],r*4+c.k);
        int64_t sum=(int64_t)prior+x*w;
        if(sum<INT32_MIN||sum>INT32_MAX) {
            log_error(cycle,"MAC","INT32 overflow pc=0x%05X k=%u neuron=%u vector=%u",c.pc,c.k,r,v);return false;
        }
        m->products_added.value[r][v]=(int32_t)sum;
    }
    return true;
fault:log_error(cycle,"MAC","invalid operands, config, accumulation or forwarding pc=0x%05X",c.pc);return false;
}
static void clock_mac(void *instance,uint64_t cycle)
{
    Mac *m=instance;Control c=Control_read(m->ports.in.control);
    if(logging_ports_enabled()) {
        log_block_begin("MAC (operand slots + 16 PEs)");
        printf("  latch=%u slot=%u field=%u compute=%u selected_slot=%u k=%u K=%u N=%u V=%u add=%u post=%u forward=%u forward_slot=%u\n",
               c.latch_en,c.latch_slot,c.latch_field,c.compute_en,c.slot,c.k,c.k_count,c.neurons,c.batch,c.accumulate,c.post_en,c.forward_en,c.forward_slot);
        log_line("memory_data",TileSignal_read(m->ports.in.response).tile);
        log_line("forward_data",ResultSignal_read(m->ports.in.forward).tile);
        printf("  sums_valid=%u\n",AccSignal_read(m->ports.out.sums).valid);
        log_tile("accumulators",m->regs.accumulators);
        for(unsigned slot=0;slot<2;slot++) for(unsigned field=0;field<4;field++) {
            char name[48];snprintf(name,sizeof name,"slot%u.field%u valid=%u",slot,field,m->regs.slot[slot].loaded[field]);
            log_line(name,m->regs.slot[slot].operand[field]);
        }
    }
    if(c.latch_en) {
        m->regs.slot[c.latch_slot].operand[c.latch_field]=TileSignal_read(m->ports.in.response).tile;
        m->regs.slot[c.latch_slot].loaded[c.latch_field]=true;
        static const char *fields[]={"X","W","B","CFG"};
        log_flow(cycle,"MAC","LATCH","slot=%u field=%s",c.latch_slot,fields[c.latch_field]);
    }
    if(c.forward_en) {
        m->regs.slot[c.forward_slot].operand[OPERAND_X]=ResultSignal_read(m->ports.in.forward).tile;
        m->regs.slot[c.forward_slot].loaded[OPERAND_X]=true;
        log_flow(cycle,"MAC","FORWARD","slot=%u N=%u V=%u",c.forward_slot,c.neurons,c.batch);
    }
    if(c.compute_en) {
        log_flow(cycle,"MAC","COMPUTE","pc=0x%05X slot=%u k=%u active=%u/16 useful_MACs_this_clock=%u",c.pc,c.slot,c.k,c.neurons*c.batch,c.neurons*c.batch);
        for(unsigned r=0;r<c.neurons;r++) for(unsigned v=0;v<c.batch;v++) {
            const OperandSlot *slot=&m->regs.slot[c.slot];
            int x=line_byte(slot->operand[OPERAND_X],v*4+c.k);
            int w=line_byte(slot->operand[OPERAND_W],r*4+c.k);
            int32_t before=(c.k==0&&!c.accumulate)?unpack_i32(slot->operand[OPERAND_B].lane[r]):m->regs.accumulators.value[r][v];
            log_flow(cycle,"MAC","PE","pc=0x%05X k=%u r=%u v=%u acc=%d x=%d w=%d product=%d prior=%d",
                     c.pc,c.k,r,v,m->products_added.value[r][v],x,w,x*w,before);
        }
        m->regs.accumulators=m->products_added;
        m->regs.valid=true;m->regs.complete=c.k+1==c.k_count;m->regs.next_k=c.k+1;
        m->regs.neurons=c.neurons;m->regs.batch=c.batch;
        m->regs.compute_cycles++;m->regs.useful_macs+=c.neurons*c.batch;
    }
    /* Post reads the published old accumulator tile before this edge clears it. */
    if(c.post_en) {m->regs.valid=false;m->regs.complete=false;}
}
const BlockOps mac_ops={.eval=evaluate,.check=check,.clock=clock_mac};
