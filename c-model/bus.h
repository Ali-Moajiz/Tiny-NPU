#ifndef BUS_H
#define BUS_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum
{
    BUS_IDLE,
    BUS_PUBLISH,
    BUS_EVAL,
    BUS_CHECK,
    BUS_CLOCK
} BusPhase;
typedef struct
{
    BusPhase phase;
    const void *active;
    uint64_t cycle;
    bool fault;
} BusContext;

typedef struct
{
    BusContext *context;
    const void *driver;
    const char *name;
    void *payload;
    size_t size;
    uint64_t driven_cycle;
} Bus;

void bus_init(Bus *, BusContext *, const char *, const void *, void *, size_t);
bool bus_drive(Bus *, const void *, size_t);
bool bus_read(const Bus *, void *, size_t);

/* Typed signal interfaces use the same checked transport. */
#define BUS_SIGNAL(Type)                                                      \
    typedef struct                                                            \
    {                                                                         \
        Bus bus;                                                              \
        Type payload;                                                         \
    } Type##Bus;                                                              \
    static inline void Type##_connect(Type##Bus *b, BusContext *ctx,          \
                                      const char *name, const void *driver)   \
    {                                                                         \
        bus_init(&b->bus, ctx, name, driver, &b->payload, sizeof b->payload); \
    }                                                                         \
    static inline void Type##_drive(Type##Bus *b, Type value)                 \
    {                                                                         \
        bus_drive(&b->bus, &value, sizeof value);                             \
    }                                                                         \
    static inline Type Type##_read(const Type##Bus *b)                        \
    {                                                                         \
        Type value = {0};                                                     \
        bus_read(&b->bus, &value, sizeof value);                              \
        return value;                                                         \
    }

#endif
