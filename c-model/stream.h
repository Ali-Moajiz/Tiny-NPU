#ifndef STREAM_H
#define STREAM_H
#include "framework.h"
#include "signals.h"
#define STREAM_CAPACITY 256u

/* Simulation endpoint for runtime streams, not a post-processing FIFO. */
typedef struct {
    Line128 input[STREAM_CAPACITY], output[STREAM_CAPACITY];
    size_t input_count, input_index, output_count;
    bool output_enabled;
    uint64_t input_available_cycle, output_available_cycle;
} StreamRegisters;

typedef struct {
    struct {
        const ControlBus *control;
        const TileSignalBus *response;
    } in;
    struct {
        TileSignalBus *input;
        StreamStatusBus *status;
    } out;
} StreamPorts;

typedef struct {
    StreamRegisters regs;
    StreamPorts ports;
} Stream;
bool stream_load_binary(Stream *, const char *path);
bool stream_save_binary(const Stream *, const char *path);
extern const BlockOps stream_ops;
#endif
