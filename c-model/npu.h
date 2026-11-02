#ifndef NPU_H
#define NPU_H
#include "controller.h"
#include "sram.h"
#include "demux.h"
#include "reg128.h"
#include "mac.h"
#include "activation.h"
#include "scale_clamp.h"
#include "fifo.h"
typedef struct {
    BusNetwork network;
    Controller controller;
    Sram sram;
    Demux demux;
    Reg128 reg_w,reg_i,reg_b,reg_cfg;
    MacArray array;
    Activation activation;
    ScaleClamp scale;
    ResultFifo fifo;
    bool error;
} Npu;
typedef enum { NPU_RUN_ERROR,NPU_RUN_COMPLETE,NPU_RUN_LIMIT } NpuRunStatus;
bool npu_init(Npu *);
bool npu_init_options(Npu *,NpuOptions);
NpuRunStatus npu_run(Npu *,uint64_t);
#endif
