#include "sram.h"
#include "sim.h"
#include <stdio.h>

/* ---- Single-port SRAM ---- */

static void publish(void *instance,uint64_t cycle)
{ (void)cycle;Sram *s=instance;TileSignal_drive(s->ports.out.response,s->regs.response); }
static bool check(void *instance,uint64_t cycle)
{
    Sram *s=instance;Control c=Control_read(s->ports.in.control);
    if((c.rd_en&&c.wr_en)||(c.rd_en&&!isa_address(c.rd_addr,0,SRAM_BYTES))||
       (c.wr_en&&!isa_address(c.wr_addr,0,SRAM_BYTES))||
       (c.rd_en&&s->regs.response.valid&&!c.read_consume)||
       (c.read_consume&&!s->regs.response.valid)) goto fault;
    if(c.wr_en) {
        if(c.write_source==WRITE_STREAM) {if(!TileSignal_read(s->ports.in.input).valid) goto fault;}
        else if(c.write_source==WRITE_RESULT) {if(!ResultSignal_read(s->ports.in.result).valid) goto fault;}
        else goto fault;
    }
    return true;
fault:log_error(cycle,"SRAM","port conflict, invalid address or invalid transfer");return false;
}
static void clock_sram(void *instance,uint64_t cycle)
{
    Sram *s=instance;Control c=Control_read(s->ports.in.control);
    if(logging_ports_enabled()) {
        log_block_begin("SRAM");
        printf("  rd_en=%u rd_addr=%05X wr_en=%u wr_addr=%05X byte_enable=%04X response_valid=%u consume=%u\n",
               c.rd_en,c.rd_addr,c.wr_en,c.wr_addr,c.byte_enable,s->regs.response.valid,c.read_consume);
        log_line("read_response",s->regs.response.tile);
        log_line("stream_write",TileSignal_read(s->ports.in.input).tile);
        log_line("post_write",ResultSignal_read(s->ports.in.result).tile);
    }
    if(c.read_consume) s->regs.response.valid=false;
    if(c.rd_en) {
        s->regs.response=(TileSignal){s->regs.memory[c.rd_addr/16],true};s->regs.reads++;
        log_flow(cycle,"SRAM","READ","address=0x%05X line=%u available_cycle=%llu",c.rd_addr,c.rd_addr/16,(unsigned long long)(cycle+1));
    }
    if(c.wr_en) {
        Line128 data=c.write_source==WRITE_STREAM?TileSignal_read(s->ports.in.input).tile:ResultSignal_read(s->ports.in.result).tile;
        for(unsigned b=0;b<16;b++) if(c.byte_enable&(1u<<b))
            line_set_byte(&s->regs.memory[c.wr_addr/16],b,(uint8_t)line_byte(data,b));
        s->regs.writes++;
        log_flow(cycle,"SRAM","WRITE","address=0x%05X byte_enable=0x%04X source=%s",c.wr_addr,c.byte_enable,c.write_source==WRITE_STREAM?"stream":"post");
    }
}
const BlockOps sram_ops={.publish=publish,.check=check,.clock=clock_sram};
