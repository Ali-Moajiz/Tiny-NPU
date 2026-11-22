#ifndef SIM_H
#define SIM_H
#include "types.h"
#include "sram.h"
/* Host-only services. Do not use this file as an HLS top/interface. */
#ifndef NPU_TB
#define NPU_TB 0
#endif

/* ---- Diagnostics ---- */

#ifndef NPU_LOG_FLOW
#define NPU_LOG_FLOW 1
#endif
#ifndef NPU_LOG_PORTS
#define NPU_LOG_PORTS 0
#endif
bool logging_ports_enabled(void);
bool logging_flow_enabled(void);
void log_flow(uint64_t cycle,const char *block,const char *event,const char *format,...);
void log_error(uint64_t cycle,const char *block,const char *format,...);
void log_cycle_begin(uint64_t cycle);
void log_block_begin(const char *name);
void log_line(const char *name, Line128 line);
void log_tile(const char *name, TileSums tile);

/* ---- Runtime stream test endpoint ---- */

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

/* ---- SRAM initialization ---- */

bool sram_load_hex(Sram *, const char *path, char *error, size_t error_size);

/* ---- Optional terminal/random testbench ---- */

#if NPU_TB
/* 1=batch supplied, 0=EOF/quit, -1=invalid input. Values are already INT8. */
int tb_supply(Stream *,unsigned batch,unsigned features,bool interactive);
void tb_print_tensor(const char *,const Line128 *,size_t,unsigned batch,unsigned features);
#endif

#endif
