#ifndef BUS_H
#define BUS_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* GENERIC simulation transport. No controller/MAC/SRAM signal names here.
 * One Bus carries any fixed-size C payload between a producer and consumers.
 * Registered values persist; valid is an event flag, expiring unless rewritten.
 * Wires carry combinational controls/data within a cycle, in traversal order.
 * Do not put pointers in payloads: memcpy copies a value, not pointed-to data.
 */
#define BUS_MAX_PAYLOAD 64
#define BUS_MAX_LINKS 32
#define BUS_MAX_MODULES 32

typedef struct
{
    unsigned char current[BUS_MAX_PAYLOAD], next[BUS_MAX_PAYLOAD];
    size_t payload_size;
    bool valid, next_valid, written, error;
    bool wire; /* immediate combinational link; producer must run before consumer */
} Bus;

bool bus_init(Bus *bus, size_t payload_size);
bool bus_init_wire(Bus *bus, size_t payload_size);
bool bus_read(const Bus *bus, void *payload, size_t payload_size);
bool bus_write(Bus *bus, const void *payload, size_t payload_size);

/* Every module registers the same callback signature. */
typedef void (*BusCycleFunction)(void *module, uint64_t cycle);
typedef struct
{
    BusCycleFunction cycle;
    void *module;
} BusModule;

typedef struct
{
    Bus *links[BUS_MAX_LINKS];
    BusModule modules[BUS_MAX_MODULES];
    size_t link_count, module_count;
    uint64_t cycle;
    bool error;
    bool logging; /* cycle headings; disabled by default */
} BusNetwork;

void bus_network_init(BusNetwork *network);
void bus_network_set_logging(BusNetwork *network, bool enabled);
bool bus_register_link(BusNetwork *network, Bus *link);
bool bus_register_module(BusNetwork *network, BusCycleFunction cycle, void *module);
/* Set execution order explicitly after initialization; must list ALL module contexts. */
bool bus_set_module_order(BusNetwork *network, void *const *modules, size_t count);
/* Cycle owner: clear wire strobes, traverse in dependency order, commit registers. */
bool bus_run_cycle(BusNetwork *network);
#endif
