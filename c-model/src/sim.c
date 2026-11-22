#include "sim.h"
#include "npu.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Diagnostics ---- */

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
    if(logging_ports_enabled() || logging_flow_enabled()) printf("\n================ CYCLE %06" PRIu64 " ================\n",cycle);
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

/* ---- SRAM image loader (initialization only) ---- */

bool sram_load_hex(Sram *s,const char *path,char *error,size_t error_size)
{
    FILE *file=fopen(path,"r");
    if(!file) {snprintf(error,error_size,"cannot open SRAM image: %s",path);return false;}
    /* Stage the image on the host heap; commit only after full validation. */
    Line128 *memory=malloc(sizeof s->regs.memory);
    if(!memory) {fclose(file);snprintf(error,error_size,"cannot allocate %u-byte SRAM staging image",SRAM_BYTES);return false;}
    unsigned count=0,line_no=0;char text[256];bool ok=true;
    while(fgets(text,sizeof text,file)) {
        line_no++;
        if(!strchr(text,'\n')&&!feof(file)) {ok=false;break;}
        char *p=text;while(isspace((unsigned char)*p)) p++;
        if(!*p||*p=='#'||(p[0]=='/'&&p[1]=='/')) continue;
        if(count>=SRAM_LINES) {ok=false;break;}
        if(strlen(p)<32) {ok=false;break;}
        Line128 word={{0}};
        for(unsigned i=0;i<32;i++) {
            unsigned char ch=(unsigned char)p[i];unsigned v;
            if(ch>='0'&&ch<='9') v=ch-'0';
            else if(ch>='a'&&ch<='f') v=ch-'a'+10;
            else if(ch>='A'&&ch<='F') v=ch-'A'+10;
            else {ok=false;break;}
            word.lane[3-i/8]=(word.lane[3-i/8]<<4)|v;
        }
        if(!ok) break;
        p+=32;while(isspace((unsigned char)*p)) p++;
        if(*p&&*p!='#'&&!(p[0]=='/'&&p[1]=='/')) {ok=false;break;}
        memory[count++]=word;
    }
    bool read_failed=ferror(file)!=0;
    int close_status=fclose(file);
    if(read_failed||close_status) {
        snprintf(error,error_size,"cannot read/close SRAM image: %s",path);
        free(memory);return false;
    }
    if(!ok||count!=SRAM_LINES) {
        snprintf(error,error_size,"invalid SRAM hex at line %u: expected exactly %u words of 32 hex digits (got %u)",line_no,SRAM_LINES,count);
        free(memory);
        return false;
    }
    memcpy(s->regs.memory,memory,sizeof s->regs.memory);
    free(memory);
    s->regs.response=(TileSignal){0};s->regs.reads=0;s->regs.writes=0;
    return true;
}

/* ---- Stream endpoint and binary files ---- */

bool stream_load_binary(Stream *s, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return false;
    unsigned char bytes[16];
    size_t n;
    s->regs.input_count = 0;
    s->regs.input_index = 0;
    bool ok = true;
    while ((n = fread(bytes, 1, 16, f)) != 0)
    {
        if (n != 16 || s->regs.input_count == STREAM_CAPACITY)
        {
            ok = false;
            break;
        }
        Line128 line = {{0}};
        for (unsigned i = 0; i < 16; i++)
            line_set_byte(&line, i, bytes[i]);
        s->regs.input[s->regs.input_count++] = line;
    }
    if (ferror(f))
        ok = false;
    fclose(f);
    return ok;
}
bool stream_save_binary(const Stream *s, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return false;
    bool ok = true;
    for (size_t b = 0; b < s->regs.output_count; b++)
        for (unsigned i = 0; i < 16; i++)
            if (fputc((uint8_t)line_byte(s->regs.output[b], i), f) == EOF)
                ok = false;
    if (fclose(f))
        ok = false;
    return ok;
}
static void publish(void *instance, uint64_t cycle)
{
    Stream *s = instance;
    bool valid = s->regs.input_index < s->regs.input_count && cycle >= s->regs.input_available_cycle;
    TileSignal data = {0};
    data.valid = valid;
    if (valid)
        data.tile = s->regs.input[s->regs.input_index];
    TileSignal_drive(s->ports.out.input, data);
    StreamStatus_drive(s->ports.out.status, (StreamStatus){valid, s->regs.output_enabled && s->regs.output_count < STREAM_CAPACITY && cycle >= s->regs.output_available_cycle});
}
static bool check(void *instance, uint64_t cycle)
{
    Stream *s = instance;
    Control c = Control_read(s->ports.in.control);
    StreamStatus status = StreamStatus_read(s->ports.out.status);
    if ((c.input_pop && !status.input_valid) || (c.output_push && (!status.output_ready || !TileSignal_read(s->ports.in.response).valid)))
    {
        log_error(cycle, "Stream", "invalid stream handshake");
        return false;
    }
    return true;
}
static void clock_stream(void *instance, uint64_t cycle)
{
    Stream *s = instance;
    Control c = Control_read(s->ports.in.control);
    if (logging_ports_enabled())
    {
        log_block_begin("Stream endpoint");
        StreamStatus st = StreamStatus_read(s->ports.out.status);
        printf("  input_valid=%u output_ready=%u input_pop=%u output_push=%u response_valid=%u\n", st.input_valid, st.output_ready, c.input_pop, c.output_push, TileSignal_read(s->ports.in.response).valid);
        log_line("input_data", TileSignal_read(s->ports.out.input).tile);
        log_line("output_data", TileSignal_read(s->ports.in.response).tile);
    }
    if (c.input_pop)
    {
        s->regs.input_index++;
        log_flow(cycle, "Stream", "INPUT", "beat=%zu", s->regs.input_index - 1);
    }
    if (c.output_push)
    {
        s->regs.output[s->regs.output_count++] = TileSignal_read(s->ports.in.response).tile;
        log_flow(cycle, "Stream", "OUTPUT", "beat=%zu", s->regs.output_count - 1);
    }
}
const BlockOps stream_ops = {.publish = publish, .check = check, .clock = clock_stream};

/* ---- Optional random/interactive input ---- */

#if NPU_TB
int tb_supply(Stream *s,unsigned batch,unsigned features,bool interactive)
{
    size_t beats=(features+3)/4;s->regs.input_count=beats;s->regs.input_index=0;
    memset(s->regs.input,0,beats*sizeof s->regs.input[0]);
    for(unsigned v=0;v<batch;v++) {
        char text[8192],*p=NULL;
        if(interactive) {
            printf("Vector %u/%u: enter %u quantized INT8 values (q to quit): ",v+1,batch,features);fflush(stdout);
            if(!fgets(text,sizeof text,stdin)) return 0;
            if(!strchr(text,'\n')&&!feof(stdin)) return -1;
            p=text;while(isspace((unsigned char)*p)) p++;
            if(*p=='q'||*p=='Q') return 0;
        }
        for(unsigned f=0;f<features;f++) {
            long value;
            if(interactive) {
                char *end;errno=0;value=strtol(p,&end,10);
                if(end==p||errno||value< -128||value>127) return -1;
                p=end;
            } else value=rand()%256-128;
            line_set_byte(&s->regs.input[f/4],v*4+f%4,(uint8_t)value);
        }
        if(interactive) {while(isspace((unsigned char)*p)) p++;if(*p) return -1;}
    }
    return 1;
}
void tb_print_tensor(const char *label,const Line128 *beats,size_t count,unsigned batch,unsigned features)
{
    printf("%s\n",label);
    if(count!=(features+3)/4) {printf("  beat count differs from requested feature count\n");return;}
    for(unsigned v=0;v<batch;v++) {
        printf("  vector %u: [",v);
        for(unsigned f=0;f<features;f++) printf("%s%d",f?", ":"",line_byte(beats[f/4],v*4+f%4));
        puts("]");
    }
}
#endif

/* ---- Metrics report ---- */

void npu_print_metrics(const Npu *n)
{
    double compute = n->mac.regs.compute_cycles ? 100.0 * (double)n->mac.regs.useful_macs / (16.0 * (double)n->mac.regs.compute_cycles) : 0;
    double whole = n->cycle ? 100.0 * (double)n->mac.regs.useful_macs / (16.0 * (double)n->cycle) : 0;
    printf("METRICS cycles=%" PRIu64 " instructions=%" PRIu64 " compute_cycles=%" PRIu64 " useful_macs=%" PRIu64 " bus_stall_cycles=%" PRIu64 " sram_reads=%" PRIu64 " sram_writes=%" PRIu64 " compute_util=%.2f%% whole_util=%.2f%%\n",
           n->cycle, n->controller.regs.instructions, n->mac.regs.compute_cycles, n->mac.regs.useful_macs, n->controller.regs.bus_stall_cycles, n->sram.regs.reads, n->sram.regs.writes, compute, whole);
}
