#include "logging.h"
#include <stdio.h>

#include <inttypes.h>
#include <stdarg.h>

bool logging_ports_enabled(void) { return NPU_LOG_PORTS!=0; }
bool logging_flow_enabled(void) { return NPU_LOG_FLOW!=0; }

void log_flow(uint64_t cycle,const char *block,const char *event,const char *format,...)
{
    if(!logging_flow_enabled()) return;
    printf("[cycle=%06" PRIu64 "] [FLOW] [%-12s] %-18s ",cycle,block,event);
    va_list args;va_start(args,format);vprintf(format,args);va_end(args);
    putchar('\n');
}
void log_error(uint64_t cycle,const char *block,const char *format,...)
{
    fprintf(stderr,"[cycle=%06" PRIu64 "] [ERROR] [%-12s] ",cycle,block);
    va_list args;va_start(args,format);vfprintf(stderr,format,args);va_end(args);
    fputc('\n',stderr);
}

void log_cycle_begin(uint64_t cycle)
{
    if(logging_ports_enabled()) printf("\n================ CYCLE %06" PRIu64 " ================\n",cycle);
}
void log_block_begin(const char *name) { if(logging_ports_enabled()) printf("\n[BLOCK] %s\n",name); }


void log_line(const char *name, Line128 line)
{
    printf("  %s=%08" PRIX32 "_%08" PRIX32 "_%08" PRIX32 "_%08" PRIX32 "\n",
           name, line.lane[3], line.lane[2], line.lane[1], line.lane[0]);
}

void log_tile(const char *name, TileSums tile)
{
    printf("  %s=", name);
    for (unsigned r = 0; r < ARRAY_SIZE; r++) {
        printf(" [%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32 "]",
               tile.value[r][0], tile.value[r][1], tile.value[r][2], tile.value[r][3]);
    }
    putchar('\n');
}
