#include "logging.h"
#include <stdio.h>
#include <string.h>
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

/* Inspect bytes without consuming events or changing any simulation state.
 * Current data is printed even if valid=0: retained bits are not a new event.
 */
static void format_value(char text[96],const unsigned char *data,size_t size,LogType type)
{
    if(type==LOG_U128 && size==16) {
        uint32_t v[4];memcpy(v,data,sizeof v);
        snprintf(text,96,"0x%08" PRIX32 "%08" PRIX32 "%08" PRIX32 "%08" PRIX32,v[3],v[2],v[1],v[0]);return;
    }
    if(type==LOG_REG128 && size==20) { /* Line128 followed by the model's 'loaded' flag */
        uint32_t v[4];memcpy(v,data,sizeof v);
        snprintf(text,96,"0x%08" PRIX32 "%08" PRIX32 "%08" PRIX32 "%08" PRIX32 " loaded=%u",v[3],v[2],v[1],v[0],(unsigned)data[16]);return;
    }
    if(type==LOG_I32X4 && size==16) {
        int32_t v[4];memcpy(v,data,sizeof v);
        snprintf(text,96,"[%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32 "]",v[0],v[1],v[2],v[3]);return;
    }
    if(type==LOG_I8X4 && size==4) {
        int8_t v[4];memcpy(v,data,sizeof v);
        snprintf(text,96,"[%d,%d,%d,%d]",v[0],v[1],v[2],v[3]);return;
    }
    if(type==LOG_BYTES) {
        size_t used=0;
        for(size_t i=0;i<size && used+3<96;i++)
            used+=(size_t)snprintf(text+used,96-used,"%02X",(unsigned)data[i]);
        return;
    }
    size_t expected=(type==LOG_U8 || type==LOG_I8)?1:
                    (type==LOG_U16 || type==LOG_OPERANDS)?2:4;
    if(size!=expected) { snprintf(text,96,"<payload size mismatch>"); return; }
    switch(type) {
    case LOG_U128: case LOG_REG128: case LOG_BYTES: case LOG_I32X4: case LOG_I8X4: snprintf(text,96,"<payload size mismatch>");break;
    case LOG_U8: {
        uint8_t value;memcpy(&value,data,sizeof(value));
        snprintf(text,96,"0x%02" PRIX8 " (%u)",value,(unsigned)value);break;
    }
    case LOG_I8: {
        int8_t value;uint8_t bits;memcpy(&value,data,1);memcpy(&bits,data,1);
        snprintf(text,96,"0x%02" PRIX8 " (%d)",bits,(int)value);break;
    }
    case LOG_U16: {
        uint16_t value;memcpy(&value,data,sizeof(value));
        snprintf(text,96,"0x%04" PRIX16 " (%u)",value,(unsigned)value);break;
    }
    case LOG_U32: {
        uint32_t value;memcpy(&value,data,sizeof(value));
        snprintf(text,96,"0x%08" PRIX32 " (%" PRIu32 ")",value,value);break;
    }
    case LOG_I32: {
        uint32_t bits;int32_t value;memcpy(&bits,data,4);memcpy(&value,data,4);
        snprintf(text,96,"0x%08" PRIX32 " (%" PRId32 ")",bits,value);break;
    }
    case LOG_OPERANDS: {
        uint16_t value;memcpy(&value,data,2);
        int x=value & 255u,w=(value>>8) & 255u;
        if(x>=128) x-=256;
        if(w>=128) w-=256;
        snprintf(text,96,"0x%04" PRIX16 " (x=%d,w=%d)",value,x,w);break;
    }
    }
}
void log_port(const char *name,const Bus *b,LogDirection direction,LogType type)
{
    if(!logging_ports_enabled()) return;
    printf("  %-3s %-27s ",direction==LOG_INPUT?"IN":"OUT",name);
    if(!b) { puts("<not connected>"); return; }
    char current[96],next[96];
    format_value(current,b->current,b->payload_size,type);
    printf("%s=%s valid=%u",b->wire?"WIRE":"CURRENT",current,(unsigned)b->valid);
    if(direction==LOG_OUTPUT && !b->wire) {
        format_value(next,b->next,b->payload_size,type);
        printf(" | NEXT=%-31s valid=%u written=%u",next,(unsigned)b->next_valid,(unsigned)b->written);
    }
    if(b->error) printf(" BUS_ERROR");
    puts("");
}
