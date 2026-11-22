#include "bus.h"
#include "sim.h"
#include <string.h>

/* ---- Checked typed wires ---- */

void bus_init(Bus *b, BusContext *ctx, const char *name, const void *driver,
              void *payload, size_t size)
{
    *b = (Bus){.context = ctx, .driver = driver, .name = name,
               .payload = payload, .size = size};
}

static bool fault(const Bus *b, const char *reason)
{
    b->context->fault = true;
    log_error(b->context->cycle, b->name, "%s", reason);
    return false;
}

bool bus_drive(Bus *b, const void *value, size_t size)
{
    BusContext *ctx = b->context;
    if ((ctx->phase != BUS_PUBLISH && ctx->phase != BUS_EVAL) ||
        ctx->active != b->driver || b->driven_cycle == ctx->cycle || size != b->size)
        return fault(b, "invalid bus drive (phase, driver, duplicate or size)");
    memcpy(b->payload, value, size);
    b->driven_cycle = ctx->cycle;
    return true;
}

bool bus_read(const Bus *b, void *value, size_t size)
{
    BusContext *ctx = b->context;
    if (ctx->phase == BUS_IDLE || ctx->phase == BUS_PUBLISH ||
        b->driven_cycle != ctx->cycle || size != b->size)
        return fault(b, "invalid bus read (phase, undriven signal or size)");
    memcpy(value, b->payload, size);
    return true;
}

/* ---- Registered block traversal ---- */

bool framework_register(Block *blocks, size_t *count, size_t capacity,
                        void *instance, const BlockOps *ops)
{
    if (!blocks || !count || !instance || !ops || *count >= capacity)
        return false;
    for (size_t i = 0; i < *count; i++)
        if (blocks[i].instance == instance)
            return false;
    blocks[(*count)++] = (Block){instance, ops};
    return true;
}

bool framework_step(BusContext *ctx, const Block *blocks, size_t count, uint64_t cycle)
{
    ctx->cycle = cycle;
    for (BusPhase phase = BUS_PUBLISH; phase <= BUS_CLOCK && !ctx->fault; phase++)
    {
        ctx->phase = phase;
        for (size_t i = 0; i < count && !ctx->fault; i++)
        {
            Block b = blocks[i];
            ctx->active = b.instance;
            switch (phase)
            {
            case BUS_PUBLISH:
                if (b.ops->publish)
                    b.ops->publish(b.instance, cycle);
                break;
            case BUS_EVAL:
                if (b.ops->eval && !b.ops->eval(b.instance, cycle))
                    ctx->fault = true;
                break;
            case BUS_CHECK:
                if (b.ops->check && !b.ops->check(b.instance, cycle))
                    ctx->fault = true;
                break;
            case BUS_CLOCK:
                if (b.ops->clock)
                    b.ops->clock(b.instance, cycle);
                break;
            case BUS_IDLE:
                break;
            }
        }
    }
    ctx->active = NULL;
    ctx->phase = BUS_IDLE;
    return !ctx->fault;
}
