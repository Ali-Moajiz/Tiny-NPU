#ifndef NPU_H
#define NPU_H
#include "controller.h"
#include "sram.h"
#include "stream.h"
#include "mac.h"
#include "activation.h"
#include "scale_clamp.h"
#define NPU_BLOCK_CAPACITY (5u + ARRAY_SIZE * ARRAY_SIZE + 2u * ARRAY_SIZE)
typedef struct
{
    BusContext bus_context;
    Controller controller;
    Sram sram;
    Stream stream;
    Mac mac;
    Activation activation;
    ScaleClamp scale;
    struct
    {
        ControlBus control;
        TileSignalBus memory, input;
        StreamStatusBus stream_status;
        AccLaneBus sums[ARRAY_SIZE], activated[ARRAY_SIZE];
        ResultSignalBus result;
    } wires;
    Block blocks[NPU_BLOCK_CAPACITY];
    size_t block_count;
    uint64_t cycle;
    bool error;
} Npu;
/* Initialize in its final address: ports point into this instance's wires. */
void npu_init(Npu *);
bool npu_step(Npu *);
bool npu_run(Npu *, uint64_t max_cycles);
void npu_print_metrics(const Npu *);
#endif
