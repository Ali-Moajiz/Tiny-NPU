/* Focused framework/extension tests that do not depend on a neural network. */
#include "npu.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static bool extension_validate(Instruction i)
{ return !i.src&&!i.dest&&!i.dims&&!i.flags; }
static unsigned extension_calls;
static ExecStatus extension_execute(const Controller *c,const ControllerInputs *in,Control *out)
{
    (void)c;(void)in;(void)out;
    extension_calls++;
    return EXEC_DONE;
}
static void extension_retire(Controller *c) { c->regs.done=true; }
static void test_registry(void)
{
    Npu n;npu_init(&n);
    static const InstructionHandler extension={9,"CUSTOM_STOP",extension_validate,extension_execute,extension_retire};
    assert(controller_register(&n.controller,&extension));
    assert(!controller_register(&n.controller,&extension));
    assert(!controller_register(&n.controller,NULL));
    InstructionHandler invalid=extension;invalid.opcode=16;
    assert(!controller_register(&n.controller,&invalid));
    invalid=extension;invalid.opcode=10;invalid.execute=NULL;
    assert(!controller_register(&n.controller,&invalid));
    n.sram.regs.memory[ENTRY_PC/16].lane[3]=0x90000000;
    assert(npu_run(&n,20));
    assert(n.cycle==5&&extension_calls==1&&n.controller.regs.instructions==1);
}
static void test_rounding(void)
{
    assert(scale_clamp_value(1,1,1)==0);
    assert(scale_clamp_value(3,1,1)==2);
    assert(scale_clamp_value(5,1,1)==2);
    assert(scale_clamp_value(-1,1,1)==0);
    assert(scale_clamp_value(-3,1,1)==-2);
    assert(scale_clamp_value(-5,1,1)==-2);
    assert(scale_clamp_value(INT32_MIN,INT32_MAX,62)==-1);
    assert(scale_clamp_value(INT32_MAX,INT32_MAX,62)==1);
    assert(scale_clamp_value(INT32_MIN,INT32_MAX,0)==-128);
    assert(scale_clamp_value(INT32_MAX,INT32_MAX,0)==127);
}
/* Standalone SRAM driver validates masked writes and atomic rejection. */
typedef struct {Control c;ControlBus *bus;} Driver;
static void drive(void *instance,uint64_t cycle)
{ (void)cycle;Driver *d=instance;Control_drive(d->bus,d->c); }
static const BlockOps driver_ops={.publish=drive};
static void test_sram_port(void)
{
    BusContext ctx={0};Sram s={0};ControlBus ctrl;TileSignalBus input,response;ResultSignalBus result;
    Driver d={.bus=&ctrl};
    Control_connect(&ctrl,&ctx,"ctrl",&d);
    TileSignal_connect(&input,&ctx,"input",&d);ResultSignal_connect(&result,&ctx,"result",&d);
    TileSignal_connect(&response,&ctx,"response",&s);
    s.ports.in.control=&ctrl;s.ports.in.input=&input;s.ports.in.result=&result;s.ports.out.response=&response;
    /* Pre-drive constant payloads for these unit-test cycles. */
    input.payload=(TileSignal){.tile={{0xAABBCCDD,0x11223344,0x55667788,0x99AABBCC}},.valid=true};
    result.payload=(ResultSignal){0};
    Block blocks[]={{&d,&driver_ops},{&s,&sram_ops}};
    d.c=(Control){.wr_en=true,.wr_addr=0x2000,.byte_enable=0x8001,.write_source=WRITE_STREAM};
    input.bus.driven_cycle=result.bus.driven_cycle=1;
    assert(framework_step(&ctx,blocks,2,1));
    assert(s.regs.memory[0x200].lane[0]==0xDD&&s.regs.memory[0x200].lane[3]==0x99000000);
    d.c=(Control){.rd_en=true,.rd_addr=0x2000};
    input.bus.driven_cycle=result.bus.driven_cycle=2;
    assert(framework_step(&ctx,blocks,2,2));
    assert(!response.payload.valid&&s.regs.response.valid); /* visible next cycle */
    d.c=(Control){0};
    input.bus.driven_cycle=result.bus.driven_cycle=3;
    assert(framework_step(&ctx,blocks,2,3));
    assert(response.payload.valid&&s.regs.response.valid); /* held without consume */
    d.c=(Control){.rd_en=true,.wr_en=true,.rd_addr=0x2000,.wr_addr=0x2000,.byte_enable=0xffff};
    input.bus.driven_cycle=result.bus.driven_cycle=4;
    assert(!framework_step(&ctx,blocks,2,4));
    assert(s.regs.reads==1&&s.regs.writes==1); /* no partial edge */
}
int main(void)
{test_registry();test_rounding();test_sram_port();puts("Core registry, rounding and SRAM port checks passed");return 0;}
