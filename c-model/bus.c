#include "bus.h"
#include "logging.h"
#include <string.h>

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
