#ifndef LOGGING_H
#define LOGGING_H
#include "datapath.h"
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
#endif
