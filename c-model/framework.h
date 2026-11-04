#ifndef FRAMEWORK_H
#define FRAMEWORK_H
#include "bus.h"

/* Publish reads local registers only. Eval and check may read settled
 * inputs. Check every active transfer before clock, where callbacks update
 * only their own registers and may not drive any bus. */
typedef struct {
    void (*publish)(void *, uint64_t);
    bool (*eval)(void *, uint64_t);
    bool (*check)(void *, uint64_t);
    void (*clock)(void *, uint64_t);
} BlockOps;

typedef struct {
    void *instance;
    const BlockOps *ops;
} Block;

bool framework_step(BusContext *, const Block *, size_t, uint64_t);
#endif
