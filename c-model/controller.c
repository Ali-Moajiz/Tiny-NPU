#include "controller.h"
#include "logging.h"
#include <stdio.h>
#include <string.h>
void controller_init(Controller *c)
{
    memset(c,0,sizeof *c);
    c->regs.pc=ENTRY_PC;
    c->regs.state=CTRL_FETCH;
    (void)controller_register_builtins(c);
}
bool controller_register(Controller *c, const InstructionHandler *h)
{
    if (!h || h->opcode>15 || !h->name || !h->validate || !h->execute || c->handlers[h->opcode])
        return false;
    c->handlers[h->opcode]=h;
    return true;
}
static bool evaluate(void *instance, uint64_t cycle)
{
    Controller *c=instance;
    Control out={0}; out.pc=c->regs.pc;
    ControllerInputs in={TileSignal_read(c->ports.in.memory), StreamStatus_read(c->ports.in.stream)};
    c->execution=EXEC_PROGRESS;
    switch(c->regs.state) {
    case CTRL_FETCH:
        if (!isa_address(c->regs.pc,ENTRY_PC,SRAM_BYTES)) goto fault;
        out.rd_en=true; out.rd_addr=c->regs.pc;
        break;
    case CTRL_RESPONSE:
        if (in.memory.valid) out.read_consume=true;
        else c->execution=EXEC_WAIT;
        break;
    case CTRL_DECODE: {
        c->decoded=instruction_decode(c->regs.ir);
        const InstructionHandler *h=c->handlers[c->decoded.opcode];
        if (!h || !h->validate(c->decoded)) goto fault;
        break;
    }
    case CTRL_DISPATCH:
        c->execution=c->active_handler->execute(c,&in,&out);
        if(c->execution==EXEC_FAULT) goto fault;
        break;
    case CTRL_UPDATE: break;
    }
    Control_drive(c->ports.out.control,out);
    return true;
fault:
    log_error(cycle,"Controller","invalid instruction/state at pc=0x%04X state=%u opcode=0x%X",c->regs.pc,c->regs.state,
              c->regs.state==CTRL_DECODE?c->decoded.opcode:c->regs.instruction.opcode);
    return false;
}
static void clock_controller(void *instance,uint64_t cycle)
{
    Controller *c=instance;
    ControllerRegisters *r=&c->regs;
    Control ctrl=Control_read(c->ports.out.control);
    static const char *names[]={"FETCH","RESPONSE","DECODE","DISPATCH","UPDATE"};
    log_flow(cycle,"Controller",names[r->state],"pc=0x%04X opcode=%s step=%u status=%u",r->pc,
             c->active_handler?c->active_handler->name:"-",r->step,c->execution);
    if(logging_ports_enabled()) {
        log_block_begin("Controller");
        printf("  pc=0x%04X state=%s step=%u mem_valid=%u input_valid=%u output_ready=%u\n",
               r->pc,names[r->state],r->step,TileSignal_read(c->ports.in.memory).valid,
               StreamStatus_read(c->ports.in.stream).input_valid,StreamStatus_read(c->ports.in.stream).output_ready);
        printf("  rd=%u rd_addr=0x%04X consume=%u wr=%u wr_addr=0x%04X byte_en=%04X source=%u\n",
               ctrl.rd_en,ctrl.rd_addr,ctrl.read_consume,ctrl.wr_en,ctrl.wr_addr,ctrl.byte_enable,ctrl.write_source);
        printf("  latch=%u slot=%u field=%u compute=%u k=%u/%u N=%u V=%u accumulate=%u post=%u forward=%u target=%u input_pop=%u output_push=%u\n",
               ctrl.latch_en,ctrl.latch_slot,ctrl.latch_field,ctrl.compute_en,ctrl.k,ctrl.k_count,ctrl.neurons,ctrl.batch,
               ctrl.accumulate,ctrl.post_en,ctrl.forward_en,ctrl.forward_slot,ctrl.input_pop,ctrl.output_push);
    }
    r->cycles++;
    if(c->execution==EXEC_WAIT) {r->bus_stall_cycles++; return;}
    switch(r->state) {
    case CTRL_FETCH:r->state=CTRL_RESPONSE;break;
    case CTRL_RESPONSE:r->ir=TileSignal_read(c->ports.in.memory).tile;r->state=CTRL_DECODE;break;
    case CTRL_DECODE:
        r->instruction=c->decoded; c->active_handler=c->handlers[c->decoded.opcode];
        r->next_pc=r->pc+16; r->step=0; r->state=CTRL_DISPATCH;break;
    case CTRL_DISPATCH:
        if(c->execution==EXEC_DONE) r->state=CTRL_UPDATE;
        else r->step++;
        break;
    case CTRL_UPDATE:
        if(c->active_handler->retire) c->active_handler->retire(c);
        r->instructions++; r->pc=r->next_pc; r->state=CTRL_FETCH;
        c->active_handler=NULL;
        break;
    }
}
const BlockOps controller_ops={.eval=evaluate,.clock=clock_controller};
