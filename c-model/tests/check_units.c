#include "npu.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* ---- check_bus_contract ---- */

typedef struct { unsigned value; } Word;
BUS_SIGNAL(Word)

typedef struct {
    unsigned reg;
    WordBus *output;
    const WordBus *input;
} Stage;

static void bus_test_publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    Stage *stage = instance;
    Word_drive(stage->output, (Word){stage->reg});
}

static bool bus_test_check(void *instance, uint64_t cycle)
{
    (void)cycle;
    Stage *stage = instance;
    return Word_read(stage->input).value != 99;
}

static void clock_stage(void *instance, uint64_t cycle)
{
    (void)cycle;
    Stage *stage = instance;
    stage->reg = Word_read(stage->input).value;
}

static const BlockOps bus_test_ops = {.publish = bus_test_publish, .check = bus_test_check, .clock = clock_stage};

static void check_bus_contract(void)
{
    BusContext ctx = {0};
    WordBus ab, ba;
    Stage a = {1, &ab, &ba}, b = {2, &ba, &ab};
    Word_connect(&ab, &ctx, "a_to_b", &a);
    Word_connect(&ba, &ctx, "b_to_a", &b);
    Block blocks[] = {{&a, &bus_test_ops}, {&b, &bus_test_ops}};
    assert(framework_step(&ctx, blocks, 2, 1));
    assert(a.reg == 2 && b.reg == 1);

    Block reversed[] = {{&b, &bus_test_ops}, {&a, &bus_test_ops}};
    assert(framework_step(&ctx, reversed, 2, 2));
    assert(a.reg == 1 && b.reg == 2);

    /* A rejected edge must leave even the earlier block unchanged. */
    a.reg = 99;
    assert(!framework_step(&ctx, blocks, 2, 3));
    assert(a.reg == 99 && b.reg == 2);

    ctx.fault = false;
    assert(!framework_step(&ctx, blocks, 1, 4)); /* undriven ba */
    assert(a.reg == 99 && b.reg == 2);

    ctx.fault = false;
    a.output = &ba; /* a cannot drive b's bus */
    assert(!framework_step(&ctx, blocks, 2, 5));
    assert(a.reg == 99 && b.reg == 2);
    return;
}

/* ---- check_core_contract ---- */

/* Focused framework/extension tests that do not depend on a neural network. */
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
static void check_core_contract(void)
{
    test_registry();
    test_rounding();
    test_sram_port();
    test_map();
    puts("Core registry, rounding, memory map and SRAM port checks passed");
    return;
}

/* ---- check_pe_contract ---- */

/* Hardware-facing contracts: shared wires, private state, atomic edges and
 * registered pipeline timing. No compiler is needed for these tests. */

typedef struct { Control value; ControlBus *bus; } PeDriver;
static void publish_driver(void *instance, uint64_t cycle)
{
    (void)cycle;
    PeDriver *d = instance;
    Control_drive(d->bus, d->value);
}
static const BlockOps pe_driver_ops = {.publish=publish_driver};
static Npu n;
static PeDriver driver;

static void setup(void)
{
    npu_init(&n);
    assert(!n.error && n.block_count == NPU_BLOCK_CAPACITY);
    /* Replace only the sequencer, keeping the production netlist/framework. */
    driver = (PeDriver){.bus=&n.wires.control};
    Control_connect(&n.wires.control, &n.bus_context, "test.control", &driver);
    n.blocks[0] = (Block){&driver, &pe_driver_ops};
    for (unsigned i=0; i<4; i++) n.mac.regs.slot[0].loaded[i] = true;
    n.mac.regs.slot[0].operand[OPERAND_CFG] = (Line128){{1,0,1,0}};
}
static bool tick(Control c)
{
    driver.value = c;
    return framework_step(&n.bus_context, n.blocks, n.block_count, ++n.cycle);
}
static Control compute(unsigned k, unsigned count, bool retain)
{
    return (Control){.pc=ENTRY_PC, .compute_en=true, .k=k, .k_count=count,
                     .neurons=4, .batch=4, .accumulate=retain};
}
static Control post(unsigned column)
{
    return (Control){.pc=ENTRY_PC+16, .post_en=true, .post_col=column, .neurons=4, .batch=4};
}
static void test_registry_and_wires(void)
{
    setup();
    for (size_t a=0; a<n.block_count; a++)
        for (size_t b=a+1; b<n.block_count; b++)
            assert(n.blocks[a].instance != n.blocks[b].instance);
    assert(!framework_register(n.blocks, &n.block_count, NPU_BLOCK_CAPACITY, &n.mac, &mac_ops));
    Block blocks[2]; size_t count=0;
    assert(framework_register(blocks, &count, 2, &n.mac, &mac_ops));
    assert(!framework_register(blocks, &count, 2, &n.mac, &mac_ops));
    assert(!framework_register(blocks, &count, 2, NULL, &mac_ops));
    for (unsigned v=0; v<4; v++) for (unsigned r=0; r<4; r++) {
        const MacPe *p = &n.mac.cell[v][r];
        assert(p->ports.in.x == &n.mac.wires.x[v]);
        assert(p->ports.in.w == &n.mac.wires.w[r]);
        assert(p->ports.in.bias == &n.mac.wires.bias[r]);
        assert(p->ports.out.sum == &n.mac.wires.sum[v][r]);
        assert(p->ports.out.sum->bus.driver == p);
    }
}
static int32_t expected[4][4];
static void calculate_tile(void)
{
    static const int8_t x[4][4] = {{1,2,3,4},{2,1,0,3},{1,0,2,1},{0,1,1,2}};
    static const int8_t w[4][4] = {{1,1,1,1},{1,-1,1,-1},{-1,1,-1,1},{2,1,-1,1}};
    static const int32_t bias[4] = {1,2,-3,4};
    OperandSlot *s = &n.mac.regs.slot[0];
    for (unsigned v=0; v<4; v++) for (unsigned k=0; k<4; k++) {
        line_set_byte(&s->operand[OPERAND_X], 4*v+k, (uint8_t)x[v][k]);
        line_set_byte(&s->operand[OPERAND_W], 4*v+k, (uint8_t)w[v][k]);
    }
    for (unsigned r=0; r<4; r++) s->operand[OPERAND_B].lane[r] = (uint32_t)bias[r];
    for (unsigned k=0; k<4; k++) {
        int32_t old[4][4];
        memcpy(old, expected, sizeof old);
        assert(tick(compute(k,4,false)));
        for (unsigned v=0; v<4; v++) for (unsigned r=0; r<4; r++) {
            expected[v][r] = (k ? expected[v][r] : bias[r]) + x[v][k]*w[r][k];
            assert(n.mac.cell[v][r].regs.accumulator == expected[v][r]);
            if (k) assert(n.mac.wires.sum[v][r].payload.value == old[v][r]);
        }
    }
    assert(n.mac.regs.useful_macs == 64 && n.mac.regs.compute_cycles == 4);
    assert(tick((Control){0}));
    for (unsigned v=0; v<4; v++) for (unsigned r=0; r<4; r++)
        assert(n.mac.cell[v][r].regs.accumulator == expected[v][r]);
}
static void test_pipeline(bool reverse_registered_lanes)
{
    setup();
    if (reverse_registered_lanes) {
        /* No same-cycle leakage between registered stages, even if their
         * evaluation order is reversed. Combinational MAC mux order stays. */
        for (size_t i=20; i<28; i+=2) {
            Block tmp=n.blocks[i]; n.blocks[i]=n.blocks[i+1]; n.blocks[i+1]=tmp;
        }
    }
    calculate_tile();
    for (unsigned step=0; step<6; step++) {
        assert(tick(step<4 ? post(step) : (Control){0}));
        assert(n.scale.regs.result.valid == (step == 5));
        for (unsigned v=0; v<4; v++) {
            assert(n.activation.lane[v].regs.output.valid == (step<4));
            assert(n.scale.lane[v].regs.output.valid == (step>=1 && step<=4));
            if (step<4) assert(n.activation.lane[v].regs.output.tag.column == step);
            if (step>=1 && step<=4) assert(n.scale.lane[v].regs.output.tag.column == step-1);
        }
        if (step>=2) assert(n.scale.regs.next_column == step-1);
    }
    for (unsigned v=0; v<4; v++) for (unsigned r=0; r<4; r++) {
        int32_t relu=expected[v][r]<0 ? 0 : expected[v][r];
        assert(line_byte(n.scale.regs.result.tile,4*v+r) == relu);
        assert(!n.mac.cell[v][r].regs.valid);
    }
    assert(!n.wires.result.payload.valid); /* Last capture not visible early. */
    assert(tick((Control){.forward_en=true,.forward_slot=1,.neurons=4,.batch=4}));
    assert(n.wires.result.payload.valid);
    assert(!memcmp(&n.mac.regs.slot[1].operand[OPERAND_X], &n.scale.regs.result.tile, sizeof(Line128)));
    assert(tick((Control){0}));
    assert(n.scale.regs.result.valid); /* Retained until next POST. */
}
static void test_retention_and_masks(void)
{
    setup(); calculate_tile();
    for (unsigned i=0; i<4; i++) {
        n.mac.regs.slot[0].operand[OPERAND_X].lane[i]=1;
        n.mac.regs.slot[0].operand[OPERAND_W].lane[i]=2;
        n.mac.regs.slot[0].operand[OPERAND_B].lane[i]=1000;
    }
    assert(tick(compute(0,1,true)));
    for (unsigned v=0; v<4; v++) for (unsigned r=0; r<4; r++)
        assert(n.mac.cell[v][r].regs.accumulator == expected[v][r]+2);
    Control c=compute(0,1,false); c.batch=1; c.neurons=2;
    assert(tick(c));
    for (unsigned v=0; v<4; v++) for (unsigned r=0; r<4; r++)
        assert(n.mac.cell[v][r].regs.accumulator == (v==0 && r<2 ? 1002 : expected[v][r]+2));
}
static void test_atomic_faults(void)
{
    for (unsigned negative=0; negative<2; negative++) {
        setup(); calculate_tile();
        for (unsigned i=0; i<4; i++) {
            n.mac.regs.slot[0].operand[OPERAND_X].lane[i]=1;
            n.mac.regs.slot[0].operand[OPERAND_W].lane[i]=negative ? 255 : 1;
            n.mac.regs.slot[0].operand[OPERAND_B].lane[i]=0;
        }
        n.mac.regs.slot[0].operand[OPERAND_B].lane[3]=negative ? (uint32_t)INT32_MIN : INT32_MAX;
        assert(!tick(compute(0,1,false)));
        assert(n.mac.regs.compute_cycles == 4 && n.mac.regs.useful_macs == 64);
        for (unsigned v=0; v<4; v++) for (unsigned r=0; r<4; r++)
            assert(n.mac.cell[v][r].regs.accumulator == expected[v][r]);
    }
    setup(); calculate_tile();
    assert(tick(post(0))); assert(tick(post(1)));
    /* Inject a corrupt column tag into an otherwise valid registered lane. */
    n.scale.lane[2].regs.output.tag.column=3;
    assert(!tick(post(2)));
    assert(n.scale.regs.next_column==0 && n.mac.regs.post_next==2);
    assert(n.activation.lane[0].regs.output.tag.column==1);
    setup(); calculate_tile();
    n.mac.cell[3][0].regs.valid=false;
    assert(!tick(post(0))); /* An incomplete column may not silently drop a lane. */
    assert(!n.scale.regs.collecting && n.mac.regs.post_next==0);
    setup(); calculate_tile();
    n.mac.cell[3][3].regs.valid=false;
    assert(!tick(compute(0,1,true)));
    assert(n.mac.regs.compute_cycles==4);
}
static void test_activation_math(void)
{
    const int32_t values[] = {INT32_MIN,-129,-7,-5,-3,-1,0,1,127,128,INT32_MAX};
    for (unsigned i=0; i<sizeof values/sizeof values[0]; i++) {
        int32_t x=values[i], y;
        assert(activation_value(x,ACT_BYPASS,0,&y) && y==x);
        assert(activation_value(x,ACT_RELU,0,&y) && y==(x<0 ? 0 : x));
        assert(activation_value(x,ACT_CLIPPED_RELU,6,&y) && y==(x<0 ? 0 : x>6 ? 6 : x));
        assert(activation_value(x,ACT_LEAKY_RELU,0,&y) && y==x);
    }
    const int32_t in[]={-1,-3,-5,-7,INT32_MIN};
    const int32_t out[]={0,-2,-2,-4,INT32_MIN/2};
    for (unsigned i=0; i<5; i++) { int32_t y; assert(activation_value(in[i],ACT_LEAKY_RELU,1,&y) && y==out[i]); }
    int32_t y=123;
    assert(activation_value(INT32_MIN,ACT_LEAKY_RELU,31,&y) && y==-1);
    assert(!activation_value(0,4,0,&y));
    assert(!activation_value(0,ACT_LEAKY_RELU,32,&y));
    assert(!activation_value(0,ACT_CLIPPED_RELU,0,&y));
    assert(!activation_value(0,ACT_RELU,1,&y));
    assert(pe_round_shift(INT64_MIN,1)==INT64_MIN/2);
    assert(pe_round_shift(INT64_MIN,62)==-2);
}
static void check_pe_contract(void)
{
    test_registry_and_wires();
    test_pipeline(false); test_pipeline(true);
    test_retention_and_masks(); test_atomic_faults(); test_activation_math();
    puts("PE shared wiring, private registers, edge timing, masks, activation math and atomic fault checks passed");
    return;
}

int main(void)
{
    check_bus_contract();
    check_core_contract();
    check_pe_contract();
    return 0;
}
