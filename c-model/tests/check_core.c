/* Focused framework/extension tests that do not depend on a neural network. */
#include "npu.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static bool extension_validate(Instruction i)
{
    return !i.src && !i.dest && !i.dims && !i.flags;
}
static unsigned extension_calls;
static ExecStatus extension_execute(const Controller *c, const ControllerInputs *in, Control *out)
{
    (void)c;
    (void)in;
    (void)out;
    extension_calls++;
    return EXEC_DONE;
}
static void extension_retire(Controller *c) { c->regs.done = true; }
static void test_registry(void)
{
    static Npu n;
    npu_init(&n);
    static const InstructionHandler extension = {9, "CUSTOM_STOP", extension_validate, extension_execute, extension_retire};
    assert(controller_register(&n.controller, &extension));
    assert(!controller_register(&n.controller, &extension));
    assert(!controller_register(&n.controller, NULL));
    InstructionHandler invalid = extension;
    invalid.opcode = 16;
    assert(!controller_register(&n.controller, &invalid));
    invalid = extension;
    invalid.opcode = 10;
    invalid.execute = NULL;
    assert(!controller_register(&n.controller, &invalid));
    n.sram.regs.memory[ENTRY_PC / 16].lane[3] = 0x90000000;
    assert(npu_run(&n, 20));
    assert(n.cycle == 5 && extension_calls == 1 && n.controller.regs.instructions == 1);
    npu_init(&n);
    n.controller.regs.pc = SRAM_BYTES - 16;
    n.sram.regs.memory[SRAM_LINES - 1].lane[3] = 0xF0000000;
    assert(npu_run(&n, 20) && n.cycle == 5 && n.sram.regs.reads == 1);
    npu_init(&n);
    n.controller.regs.pc = SRAM_BYTES;
    assert(!npu_run(&n, 20) && !n.sram.regs.reads);
}
static void test_rounding(void)
{
    assert(scale_clamp_value(1, 1, 1) == 0);
    assert(scale_clamp_value(3, 1, 1) == 2);
    assert(scale_clamp_value(5, 1, 1) == 2);
    assert(scale_clamp_value(-1, 1, 1) == 0);
    assert(scale_clamp_value(-3, 1, 1) == -2);
    assert(scale_clamp_value(-5, 1, 1) == -2);
    assert(scale_clamp_value(INT32_MIN, INT32_MAX, 62) == -1);
    assert(scale_clamp_value(INT32_MAX, INT32_MAX, 62) == 1);
    assert(scale_clamp_value(INT32_MIN, INT32_MAX, 0) == -128);
    assert(scale_clamp_value(INT32_MAX, INT32_MAX, 0) == 127);
}
/* Standalone SRAM driver validates masked writes and atomic rejection. */
typedef struct
{
    Control c;
    ControlBus *bus;
} Driver;
static void drive(void *instance, uint64_t cycle)
{
    (void)cycle;
    Driver *d = instance;
    Control_drive(d->bus, d->c);
}
static const BlockOps driver_ops = {.publish = drive};
static void test_sram_port(void)
{
    BusContext ctx = {0};
    static Sram s;
    ControlBus ctrl;
    TileSignalBus input, response;
    ResultSignalBus result;
    Driver d = {.bus = &ctrl};
    Control_connect(&ctrl, &ctx, "ctrl", &d);
    TileSignal_connect(&input, &ctx, "input", &d);
    ResultSignal_connect(&result, &ctx, "result", &d);
    TileSignal_connect(&response, &ctx, "response", &s);
    s.ports.in.control = &ctrl;
    s.ports.in.input = &input;
    s.ports.in.result = &result;
    s.ports.out.response = &response;
    /* Pre-drive constant payloads for these unit-test cycles. */
    input.payload = (TileSignal){.tile = {{0xAABBCCDD, 0x11223344, 0x55667788, 0x99AABBCC}}, .valid = true};
    result.payload = (ResultSignal){0};
    Block blocks[] = {{&d, &driver_ops}, {&s, &sram_ops}};
    d.c = (Control){.wr_en = true, .wr_addr = ACTIVATION_BASE, .byte_enable = 0x8001, .write_source = WRITE_STREAM};
    input.bus.driven_cycle = result.bus.driven_cycle = 1;
    assert(framework_step(&ctx, blocks, 2, 1));
    assert(s.regs.memory[ACTIVATION_BASE / 16].lane[0] == 0xDD && s.regs.memory[ACTIVATION_BASE / 16].lane[3] == 0x99000000);
    d.c = (Control){.rd_en = true, .rd_addr = ACTIVATION_BASE};
    input.bus.driven_cycle = result.bus.driven_cycle = 2;
    assert(framework_step(&ctx, blocks, 2, 2));
    assert(!response.payload.valid && s.regs.response.valid); /* visible next cycle */
    d.c = (Control){0};
    input.bus.driven_cycle = result.bus.driven_cycle = 3;
    assert(framework_step(&ctx, blocks, 2, 3));
    assert(response.payload.valid && s.regs.response.valid); /* held without consume */
    d.c = (Control){.rd_en = true, .wr_en = true, .rd_addr = ACTIVATION_BASE, .wr_addr = ACTIVATION_BASE, .byte_enable = 0xffff};
    input.bus.driven_cycle = result.bus.driven_cycle = 4;
    assert(!framework_step(&ctx, blocks, 2, 4));
    assert(s.regs.reads == 1 && s.regs.writes == 1); /* no partial edge */
    /* Last physical beat, above the old capacity; preserve read latency. */
    ctx.fault = false;
    d.c = (Control){.wr_en = true, .wr_addr = SRAM_BYTES - 16, .byte_enable = 0xffff, .write_source = WRITE_STREAM};
    input.bus.driven_cycle = result.bus.driven_cycle = 5;
    assert(framework_step(&ctx, blocks, 2, 5));
    assert(!memcmp(&s.regs.memory[SRAM_LINES - 1], &input.payload.tile, sizeof(Line128)));
    d.c = (Control){.rd_en = true, .rd_addr = SRAM_BYTES - 16, .read_consume = true};
    input.bus.driven_cycle = result.bus.driven_cycle = 6;
    assert(framework_step(&ctx, blocks, 2, 6));
    assert(!memcmp(&s.regs.response.tile, &input.payload.tile, sizeof(Line128)));
    d.c = (Control){.wr_en = true, .wr_addr = SRAM_BYTES, .byte_enable = 0xffff, .write_source = WRITE_STREAM};
    input.bus.driven_cycle = result.bus.driven_cycle = 7;
    assert(!framework_step(&ctx, blocks, 2, 7));
    assert(s.regs.writes == 2 && s.regs.reads == 2);
}
static void test_map(void)
{
    const uint32_t starts[] = {WEIGHT_BASE, ACTIVATION_BASE, BIAS_BASE, CONFIG_BASE, ENTRY_PC};
    const uint32_t ends[] = {WEIGHT_END, ACTIVATION_END, BIAS_END, CONFIG_END, SRAM_BYTES};
    Controller c;
    controller_init(&c);
    uint32_t total = 0;
    for (unsigned region = 0; region < 5; region++)
    {
        uint32_t start = starts[region], end = ends[region];
        assert(!(start & 15) && !(end & 15) && end > start);
        assert(start == (region ? ends[region - 1] : 0));
        total += end - start;
        assert(isa_address(start, start, end) && isa_address(end - 16, start, end));
        assert(!isa_address(end, start, end) && !isa_address(start + 1, start, end));
        assert(!isa_address(end - 15, start, end));
        if (region < 4)
        {
            unsigned field = region == 0 ? OPERAND_W : region == 1 ? OPERAND_X
                                                   : region == 2   ? OPERAND_B
                                                                   : OPERAND_CFG;
            Instruction i = {.opcode = OP_LOAD_OPERAND, .src = start, .dest = field, .dims = 16};
            assert(c.handlers[i.opcode]->validate(i));
            i.src = end - 16;
            assert(c.handlers[i.opcode]->validate(i));
            i.src = end;
            assert(!c.handlers[i.opcode]->validate(i));
            i.src = start + 1;
            assert(!c.handlers[i.opcode]->validate(i));
        }
    }
    assert(total == SRAM_BYTES && SRAM_LINES == 65536 && c.regs.pc == CONFIG_END);
    const unsigned ops[] = {OP_STREAM_LOAD, OP_STREAM_STORE, OP_POST_PROC, OP_LOOP_START, OP_LOOP_END};
    for (unsigned op = 0; op < sizeof ops / sizeof ops[0]; op++)
    {
        bool loop = ops[op] == OP_LOOP_START || ops[op] == OP_LOOP_END;
        uint32_t start = loop ? ENTRY_PC : ACTIVATION_BASE, end = loop ? SRAM_BYTES : ACTIVATION_END;
        Instruction i = {.opcode = ops[op], .dims = loop ? 0 : ops[op] == OP_POST_PROC ? 0x11
                                                                                       : 16,
                         .flags = ops[op] == OP_POST_PROC ? 4 : 0,
                         .src = ops[op] == OP_LOOP_START ? 1 : 0};
        uint32_t *addr = ops[op] == OP_STREAM_STORE ? &i.src : &i.dest;
        *addr = start;
        assert(c.handlers[i.opcode]->validate(i));
        *addr = end - 16;
        assert(c.handlers[i.opcode]->validate(i));
        *addr = end;
        assert(!c.handlers[i.opcode]->validate(i));
        *addr = start + 1;
        assert(!c.handlers[i.opcode]->validate(i));
    }
}
int main(void)
{
    test_registry();
    test_rounding();
    test_sram_port();
    test_map();
    puts("Core registry, rounding, memory map and SRAM port checks passed");
    return 0;
}
