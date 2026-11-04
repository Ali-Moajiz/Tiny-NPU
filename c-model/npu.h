#ifndef NPU_H
#define NPU_H
#include "controller.h"
#include "sram.h"
#include "reg128.h"
#include "mac.h"
#include "scale_clamp.h"

typedef struct {
    Sram sram;
    Reg128 reg_w, reg_i, reg_b, reg_cfg;
    MacArray array;
    PostProcess post;
} DataPath;

typedef struct {
    ControlBus control;
    TileSignalBus sram_response, weights, inputs, biases, config, post_result;
    AccSignalBus mac_sums;
} NpuBuses;

/* Bus endpoints point into this instance. Initialize it at its final address. */
typedef struct {
    uint64_t cycle;
    BusContext bus_context;
    NpuBuses buses;
    Controller controller;
    DataPath data;
    bool error;
} Npu;

typedef enum { NPU_RUN_ERROR,NPU_RUN_COMPLETE,NPU_RUN_LIMIT } NpuRunStatus;
bool npu_init(Npu *);
bool npu_init_options(Npu *, NpuOptions);
bool npu_step(Npu *);
NpuRunStatus npu_run(Npu *, uint64_t);
#endif
