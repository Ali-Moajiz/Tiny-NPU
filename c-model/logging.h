#ifndef LOGGING_H
#define LOGGING_H
#include "bus.h"
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
/* Generic presentation helpers: no module or signal names are embedded here. */
typedef enum { LOG_INPUT, LOG_OUTPUT } LogDirection;
typedef enum { LOG_U8, LOG_U16, LOG_U32, LOG_I8, LOG_I32, LOG_OPERANDS, LOG_BYTES, LOG_I32X4, LOG_I8X4, LOG_U128, LOG_REG128, LOG_I32X16 } LogType;
void log_cycle_begin(uint64_t cycle);
void log_block_begin(const char *name);
void log_port(const char *name,const Bus *bus,LogDirection direction,LogType type);
#endif
