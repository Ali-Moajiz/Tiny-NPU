/* Host runner: file I/O, terminal I/O and test generation never enter the controller. */
#include "npu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef NPU_TB
#define NPU_TB 0
#endif
#ifndef NPU_BATCH
#define NPU_BATCH 1
#endif
#if NPU_TB
#include "tb.h"
#endif
static bool number(const char *text,uint64_t *out)
{
    if(!*text||*text=='-') return false;
    errno=0;char *end;unsigned long long value=strtoull(text,&end,10);
    if(errno||*end) return false;
    *out=value;return true;
}
static void usage(const char *name)
{
    printf("Usage: %s --sram sram.hex --input-bin input.bin [--output-bin output.bin] [--batch 1..4] [--max-cycles N]\n",name);
#if NPU_TB
    puts("TB: omit --input-bin for random inputs; --runs N (default 100) --seed N (42) --input-features N --output-features N; --interactive to prompt.");
#endif
}
int main(int argc,char **argv)
{
    const char *image=NULL,*input=NULL,*output=NULL;
    uint64_t batch=NPU_BATCH,max_cycles=100000,runs=1,seed=42,features=4,outputs=4;
    uint64_t input_delay=0,output_delay=0;bool interactive=false;
#if NPU_TB
    runs=100;
#endif
    for(int i=1;i<argc;i++) {
        const char *arg=argv[i];
        if(!strcmp(arg,"--help")) {usage(argv[0]);return 0;}
#if NPU_TB
        if(!strcmp(arg,"--interactive")) {interactive=true;continue;}
#endif
        if(i+1==argc) {usage(argv[0]);return 2;}
        const char *value=argv[++i];uint64_t *target=NULL;
        if(!strcmp(arg,"--sram")) image=value;
        else if(!strcmp(arg,"--input-bin")) input=value;
        else if(!strcmp(arg,"--output-bin")) output=value;
        else if(!strcmp(arg,"--batch")) target=&batch;
        else if(!strcmp(arg,"--max-cycles")) target=&max_cycles;
        else if(!strcmp(arg,"--input-delay")) target=&input_delay;
        else if(!strcmp(arg,"--output-delay")) target=&output_delay;
#if NPU_TB
        else if(!strcmp(arg,"--runs")) target=&runs;
        else if(!strcmp(arg,"--seed")) target=&seed;
        else if(!strcmp(arg,"--input-features")) target=&features;
        else if(!strcmp(arg,"--output-features")) target=&outputs;
#endif
        else {fprintf(stderr,"Unknown option: %s\n",arg);return 2;}
        if(target&&!number(value,target)) {fprintf(stderr,"Invalid number: %s\n",value);return 2;}
    }
    if(!image||batch<1||batch>4||!max_cycles||!runs||features<1||features>1024||outputs<1||outputs>1024||seed>UINT32_MAX||(!NPU_TB&&!input)||(input&&interactive)) {
        usage(argv[0]);return 2;
    }
    if(input) runs=1;
    if(output&&runs>1) {fprintf(stderr,"--output-bin requires one run (--runs 1)\n");return 2;}
    /* Stable final address: internal ports point into n.wires. Keep the large
     * instance and reset snapshot off the host stack. */
    static Npu n;npu_init(&n);char error[256];
    if(!sram_load_hex(&n.sram,image,error,sizeof error)) {fprintf(stderr,"%s\n",error);return 1;}
    static Line128 initial_image[SRAM_LINES];memcpy(initial_image,n.sram.regs.memory,sizeof initial_image);
    srand((unsigned)seed);
    for(uint64_t run=0;run<runs;run++) {
        if(run) {npu_init(&n);memcpy(n.sram.regs.memory,initial_image,sizeof initial_image);}
        n.controller.expected_batch=(unsigned)batch;
        n.stream.regs.input_available_cycle=input_delay;
        n.stream.regs.output_available_cycle=output_delay;
        if(input) {
            if(!stream_load_binary(&n.stream,input)) {fprintf(stderr,"Cannot load input: expected at most 256 complete 16-byte beats\n");return 1;}
        }
#if NPU_TB
        else {
            int status=tb_supply(&n.stream,(unsigned)batch,(unsigned)features,interactive);
            if(!status) break;
            if(status<0) {fprintf(stderr,"Invalid input: provide exactly the requested number of INT8 values\n");return 1;}
            tb_print_tensor("Input (quantized INT8):",n.stream.regs.input,n.stream.regs.input_count,(unsigned)batch,(unsigned)features);
        }
#else
        (void)interactive;(void)features;(void)outputs;
#endif
        if(!npu_run(&n,max_cycles)) return 1;
        if(n.stream.regs.input_index!=n.stream.regs.input_count) {fprintf(stderr,"HALT left unused input beats\n");return 1;}
        printf("========== Inference %llu: RESULT ==========\n",(unsigned long long)(run+1));
        for(size_t b=0;b<n.stream.regs.output_count;b++) {
            Line128 x=n.stream.regs.output[b];
            printf("OUTPUT beat=%zu hex=%08X%08X%08X%08X",b,x.lane[3],x.lane[2],x.lane[1],x.lane[0]);
            putchar('\n');
        }
#if NPU_TB
        if(!input) tb_print_tensor("Predicted output (quantized INT8):",n.stream.regs.output,n.stream.regs.output_count,(unsigned)batch,(unsigned)outputs);
#endif
        npu_print_metrics(&n);
        if(output&&!stream_save_binary(&n.stream,output)) {fprintf(stderr,"Cannot write output binary\n");return 1;}
        fflush(stdout);
    }
    return 0;
}
