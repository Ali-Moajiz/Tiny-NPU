#include "bus.h"
#include "logging.h"
#include <string.h>

bool bus_init(Bus *b, size_t size)
{
    if (!b || size == 0 || size > BUS_MAX_PAYLOAD)
        return false;
    *b = (Bus){0};
    b->payload_size = size;
    return true;
}

bool bus_init_wire(Bus *b, size_t size)
{
    if (!bus_init(b, size))
        return false;
    b->wire = true;
    return true;
}

bool bus_read(const Bus *b, void *payload, size_t size)
{
    if (!b || !payload || size != b->payload_size || !b->valid)
        return false;
    memcpy(payload, b->current, size); /* read never consumes an event */
    return true;
}

bool bus_write(Bus *b, const void *payload, size_t size)
{
    if (!b)
        return false;
    if (!payload || size != b->payload_size || b->written)
    {
        b->error = true;
        return false; /* reject size mismatch / two writes */
    }
    if (b->wire)
    {
        memcpy(b->current, payload, size);
        b->valid = true;
    }
    memcpy(b->next, payload, size); /* registered links wait until commit; wires are immediate */
    b->next_valid = true;
    b->written = true;
    return true;
}

void bus_network_init(BusNetwork *n)
{
    if (n)
        *n = (BusNetwork){0};
}

void bus_network_set_logging(BusNetwork *n, bool enabled)
{
    if (n)
        n->logging = enabled;
}

bool bus_register_link(BusNetwork *n, Bus *b)
{
    if (!n || !b || b->payload_size == 0 || n->link_count == BUS_MAX_LINKS)
        return false;
    for (size_t i = 0; i < n->link_count; i++)
        if (n->links[i] == b)
            return false;
    n->links[n->link_count++] = b;
    return true;
}

bool bus_register_module(BusNetwork *n, BusCycleFunction fn, void *module)
{
    if (!n || !fn || !module || n->module_count == BUS_MAX_MODULES)
        return false;
    for (size_t i = 0; i < n->module_count; i++)
        if (n->modules[i].cycle == fn && n->modules[i].module == module)
            return false;
    n->modules[n->module_count++] = (BusModule){fn, module};
    return true;
}

bool bus_set_module_order(BusNetwork *n, void *const *modules, size_t count)
{
    if (!n || !modules || n->error || count != n->module_count || count > BUS_MAX_MODULES)
        return false;
    BusModule ordered[BUS_MAX_MODULES];
    for (size_t i = 0; i < count; i++)
    {
        for (size_t j = 0; j < i; j++)
            if (modules[i] == modules[j])
                return false;
        bool found = false;
        for (size_t j = 0; j < n->module_count; j++)
            if (n->modules[j].module == modules[i])
            {
                ordered[i] = n->modules[j];
                found = true;
                break;
            }
        if (!found)
            return false;
    }
    for (size_t i = 0; i < count; i++)
        n->modules[i] = ordered[i];
    return true;
}

bool bus_run_cycle(BusNetwork *n)
{
    if (!n || n->error)
        return false;
    /* 1. Preserve registered snapshots; clear combinational valid strobes. */
    for (size_t i = 0; i < n->link_count; i++)
    {
        Bus *b = n->links[i];
        if (b->error)
        {
            n->error = true;
            return false;
        }
        memcpy(b->next, b->current, b->payload_size);
        b->next_valid = false;
        b->written = false;
        if (b->wire)
            b->valid = false;
    }
    if (n->logging)
        log_cycle_begin(n->cycle);
    /* 2. Every registered block executes once, with private persistent state. */
    for (size_t i = 0; i < n->module_count; i++)
        n->modules[i].cycle(n->modules[i].module, n->cycle);
    /* Abort publication on any bus error. Treat it as a fatal simulation error:
     * private module states may have advanced, so do not retry this cycle. */
    for (size_t i = 0; i < n->link_count; i++)
        if (n->links[i]->error)
        {
            n->error = true;
            return false;
        }
    /* 3. Commit registered links together. Wires were evaluated during traversal. */
    for (size_t i = 0; i < n->link_count; i++)
    {
        Bus *b = n->links[i];
        if (!b->wire)
        {
            memcpy(b->current, b->next, b->payload_size);
            b->valid = b->next_valid;
        }
    }
    n->cycle++;
    return true;
}
