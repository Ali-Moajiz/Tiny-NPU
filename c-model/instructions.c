/* ISA-specific callbacks. Add a descriptor to builtins[] to extend the ISA;
 * controller.c, SRAM, and the cycle framework do not need an opcode switch. */
#include "controller.h"
static bool dimensions(Instruction i,bool mac)
{
    unsigned n=instruction_n(i),v=instruction_v(i),k=instruction_k(i);
    return !(i.dims>>12)&&n>=1&&n<=4&&v>=1&&v<=4&&(mac?(k>=1&&k<=4):k==0);
}
static bool validate_stream_load(Instruction i)
{ return !i.src&&!i.flags&&i.dims==16&&isa_address(i.dest,0x2000,0x3000); }
static bool validate_stream_store(Instruction i)
{ return !i.dest&&!i.flags&&i.dims==16&&isa_address(i.src,0x2000,0x3000); }
static bool validate_load_operand(Instruction i)
{
    static const uint32_t begin[]={0x2000,0,0x3000,0x3800},end[]={0x3000,0x2000,0x3800,0x3900};
    return i.dest<=7&&!i.flags&&i.dims==16&&isa_address(i.src,begin[i.dest&3],end[i.dest&3]);
}
static bool validate_mac(Instruction i)
{ return dimensions(i,true)&&!i.src&&!i.dest&&!(i.flags&~3u); }
static bool validate_post(Instruction i)
{ return dimensions(i,false)&&!i.src&&!(i.flags&~5u)&&((i.flags&4)?isa_address(i.dest,0x2000,0x3000):!i.dest); }
static bool validate_forward(Instruction i)
{ return dimensions(i,false)&&!i.src&&i.dest<2&&!i.flags; }
static bool validate_loop_start(Instruction i)
{ return i.src&&!i.dims&&!i.flags&&isa_address(i.dest,ENTRY_PC,SRAM_BYTES); }
static bool validate_loop_end(Instruction i)
{ return !i.src&&!i.dims&&!i.flags&&isa_address(i.dest,ENTRY_PC,SRAM_BYTES); }
static bool validate_halt(Instruction i)
{ return !i.src&&!i.dest&&!i.dims&&!i.flags; }
static void tile_control(const Controller *c,Control *out)
{
    Instruction i=c->regs.instruction;
    out->slot=i.flags&1;out->neurons=instruction_n(i);out->batch=instruction_v(i);
}
static bool batch_matches(const Controller *c,Control out)
{ return !c->expected_batch||c->expected_batch==out.batch; }
static ExecStatus stream_load(const Controller *c,const ControllerInputs *in,Control *out)
{
    if(!in->stream.input_valid) return EXEC_WAIT;
    out->wr_en=true;out->wr_addr=c->regs.instruction.dest;out->byte_enable=0xffff;
    out->write_source=WRITE_STREAM;out->input_pop=true;
    return EXEC_DONE;
}
static ExecStatus load_operand(const Controller *c,const ControllerInputs *in,Control *out)
{
    Instruction i=c->regs.instruction;
    if(!c->regs.step) {out->rd_en=true;out->rd_addr=i.src;return EXEC_PROGRESS;}
    if(!in->memory.valid) return EXEC_WAIT;
    out->read_consume=true;out->latch_en=true;out->latch_slot=i.dest>>2;out->latch_field=i.dest&3;
    return EXEC_DONE;
}
static ExecStatus exec_mac(const Controller *c,const ControllerInputs *in,Control *out)
{
    (void)in;tile_control(c,out);
    if(!batch_matches(c,*out)) return EXEC_FAULT;
    out->compute_en=true;out->accumulate=(c->regs.instruction.flags&2)!=0;
    out->k=c->regs.step;out->k_count=instruction_k(c->regs.instruction);
    return out->k+1==out->k_count?EXEC_DONE:EXEC_PROGRESS;
}
static ExecStatus post_proc(const Controller *c,const ControllerInputs *in,Control *out)
{
    (void)in;tile_control(c,out);
    if(!batch_matches(c,*out)) return EXEC_FAULT;
    out->post_en=c->regs.step==0;
#if POST_LATENCY == 1
    if(!c->regs.step) return EXEC_PROGRESS;
#endif
    if(c->regs.instruction.flags&4) {
        out->wr_en=true;out->wr_addr=c->regs.instruction.dest;
        out->byte_enable=0xffff;out->write_source=WRITE_RESULT;
    }
    return EXEC_DONE;
}
static ExecStatus forward(const Controller *c,const ControllerInputs *in,Control *out)
{
    (void)in;tile_control(c,out);
    if(!batch_matches(c,*out)) return EXEC_FAULT;
    out->forward_en=true;out->forward_slot=c->regs.instruction.dest;return EXEC_DONE;
}
static ExecStatus stream_store(const Controller *c,const ControllerInputs *in,Control *out)
{
    if(!c->regs.step) {out->rd_en=true;out->rd_addr=c->regs.instruction.src;return EXEC_PROGRESS;}
    if(!in->memory.valid||!in->stream.output_ready) return EXEC_WAIT;
    out->read_consume=true;out->output_push=true;return EXEC_DONE;
}
static ExecStatus loop_start(const Controller *c,const ControllerInputs *in,Control *out)
{
    (void)in;(void)out;
    return c->regs.loop_active||c->regs.instruction.dest!=c->regs.pc+16?EXEC_FAULT:EXEC_DONE;
}
static void retire_loop_start(Controller *c)
{
    c->regs.loop_active=true;c->regs.loop_remaining=c->regs.instruction.src;
    c->regs.loop_pc=c->regs.instruction.dest;
}
static ExecStatus loop_end(const Controller *c,const ControllerInputs *in,Control *out)
{
    (void)in;(void)out;
    return !c->regs.loop_active||c->regs.loop_pc!=c->regs.instruction.dest?EXEC_FAULT:EXEC_DONE;
}
static void retire_loop_end(Controller *c)
{
    if(--c->regs.loop_remaining) c->regs.next_pc=c->regs.loop_pc;
    else c->regs.loop_active=false;
}
static ExecStatus halt(const Controller *c,const ControllerInputs *in,Control *out)
{ (void)in;(void)out;return c->regs.loop_active?EXEC_FAULT:EXEC_DONE; }
static void retire_halt(Controller *c) {c->regs.done=true;}
bool controller_register_builtins(Controller *c)
{
    static const InstructionHandler builtins[]={
        {OP_STREAM_LOAD,"STREAM_LOAD",validate_stream_load,stream_load,NULL},
        {OP_EXEC_MAC,"EXEC_MAC",validate_mac,exec_mac,NULL},
        {OP_POST_PROC,"POST_PROC",validate_post,post_proc,NULL},
        {OP_FORWARD,"FORWARD",validate_forward,forward,NULL},
        {OP_STREAM_STORE,"STREAM_STORE",validate_stream_store,stream_store,NULL},
        {OP_LOOP_START,"LOOP_START",validate_loop_start,loop_start,retire_loop_start},
        {OP_LOOP_END,"LOOP_END",validate_loop_end,loop_end,retire_loop_end},
        {OP_LOAD_OPERAND,"LOAD_OPERAND",validate_load_operand,load_operand,NULL},
        {OP_HALT,"HALT",validate_halt,halt,retire_halt}
    };
    for(size_t i=0;i<sizeof builtins/sizeof builtins[0];i++)
        if(!controller_register(c,&builtins[i])) return false;
    return true;
}
