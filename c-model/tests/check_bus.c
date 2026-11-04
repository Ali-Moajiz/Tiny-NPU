#include "framework.h"
#include <assert.h>

typedef struct { unsigned value; } Word;
BUS_SIGNAL(Word)

typedef struct {
    unsigned reg;
    WordBus *output;
    const WordBus *input;
} Stage;

static void publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    Stage *stage = instance;
    Word_drive(stage->output, (Word){stage->reg});
}

static bool check(void *instance, uint64_t cycle)
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

static const BlockOps ops = {.publish = publish, .check = check, .clock = clock_stage};

int main(void)
{
    BusContext ctx = {0};
    WordBus ab, ba;
    Stage a = {1, &ab, &ba}, b = {2, &ba, &ab};
    Word_connect(&ab, &ctx, "a_to_b", &a);
    Word_connect(&ba, &ctx, "b_to_a", &b);
    Block blocks[] = {{&a, &ops}, {&b, &ops}};
    assert(framework_step(&ctx, blocks, 2, 1));
    assert(a.reg == 2 && b.reg == 1);

    Block reversed[] = {{&b, &ops}, {&a, &ops}};
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
    return 0;
}
