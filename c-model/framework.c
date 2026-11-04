#include "framework.h"

bool framework_step(BusContext *ctx, const Block *blocks, size_t count, uint64_t cycle)
{
    ctx->cycle = cycle;
    for (BusPhase phase = BUS_PUBLISH; phase <= BUS_CLOCK && !ctx->fault; phase++) {
        ctx->phase = phase;
        for (size_t i = 0; i < count && !ctx->fault; i++) {
            Block b = blocks[i];
            ctx->active = b.instance;
            switch (phase) {
            case BUS_PUBLISH: if (b.ops->publish) b.ops->publish(b.instance, cycle); break;
            case BUS_EVAL: if (b.ops->eval && !b.ops->eval(b.instance, cycle)) ctx->fault = true; break;
            case BUS_CHECK: if (b.ops->check && !b.ops->check(b.instance, cycle)) ctx->fault = true; break;
            case BUS_CLOCK: if (b.ops->clock) b.ops->clock(b.instance, cycle); break;
            case BUS_IDLE: break;
            }
        }
    }
    ctx->active = NULL;
    ctx->phase = BUS_IDLE;
    return !ctx->fault;
}
