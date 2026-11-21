/* Hardware-facing contracts: shared wires, private state, atomic edges and
 * registered pipeline timing. No compiler is needed for these tests. */
#include "npu.h"
#include "pe_math.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct { Control value; ControlBus *bus; } PeDriver;
static void publish_driver(void *instance, uint64_t cycle)
{
    (void)cycle;
    PeDriver *d = instance;
    Control_drive(d->bus, d->value);
}
static const BlockOps driver_ops = {.publish=publish_driver};
static Npu n;
static PeDriver driver;

static void setup(void)
{
    npu_init(&n);
    assert(!n.error && n.block_count == NPU_BLOCK_CAPACITY);
    /* Replace only the sequencer, keeping the production netlist/framework. */
    driver = (PeDriver){.bus=&n.wires.control};
    Control_connect(&n.wires.control, &n.bus_context, "test.control", &driver);
    n.blocks[0] = (Block){&driver, &driver_ops};
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
int main(void)
{
    test_registry_and_wires();
    test_pipeline(false); test_pipeline(true);
    test_retention_and_masks(); test_atomic_faults(); test_activation_math();
    puts("PE shared wiring, private registers, edge timing, masks, activation math and atomic fault checks passed");
    return 0;
}
