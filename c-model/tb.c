#include "tb.h"
#include "npu.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { TB_INPUT, TB_QUIT, TB_INVALID, TB_IO_ERROR } TbInputStatus;

/* Read one complete line, so invalid input cannot spill into the next vector. */
static TbInputStatus tb_read_input(int8_t input[ARRAY_SIZE])
{
    char line[512];
    if(!fgets(line,sizeof line,stdin)) return ferror(stdin)?TB_IO_ERROR:TB_QUIT;
    if(!strchr(line,'\n') && !feof(stdin)) {
        int ch=getchar();
        if(ch!=EOF) {
            while(ch!='\n' && ch!=EOF) ch=getchar();
            return ferror(stdin)?TB_IO_ERROR:TB_INVALID;
        }
        if(ferror(stdin)) return TB_IO_ERROR;
    }
    char *p=line;
    while(isspace((unsigned char)*p)) p++;
    if(*p=='q' || *p=='Q') {
        p++;
        while(isspace((unsigned char)*p)) p++;
        return *p?TB_INVALID:TB_QUIT;
    }
    for(unsigned i=0;i<ARRAY_SIZE;i++) {
        while(isspace((unsigned char)*p)) p++;
        char *end;
        errno=0;
        long value=strtol(p,&end,10);
        if(p==end || errno==ERANGE || value<INT8_MIN || value>INT8_MAX ||
           (*end && !isspace((unsigned char)*end))) return TB_INVALID;
        input[i]=(int8_t)value;
        p=end;
    }
    while(isspace((unsigned char)*p)) p++;
    return *p?TB_INVALID:TB_INPUT;
}

static void tb_print_vector(const int8_t values[ARRAY_SIZE])
{
    printf("[%d, %d, %d, %d]",values[0],values[1],values[2],values[3]);
}

static int tb_infer(const int8_t input[ARRAY_SIZE],uint64_t number)
{
    /* Fresh hardware state per input; no initialized Npu object is copied.
     * Parameters are reloaded by sram_init, then the host replaces only x.
     * No clocks run during terminal input or host SRAM preloading. */
    Npu system;
    printf("\n========== Inference %" PRIu64 " : START ==========\n",number);
    if(!npu_init(&system)) {
        fprintf(stderr,"[TB] NPU initialization failed.\n");return 1;
    }
    uint32_t input_address=system.controller.layers[0].input_base;
    uint32_t output_address=system.controller.layers[NETWORK_LAYERS-1].output_base;
    if(!sram_address_valid(input_address) || !sram_address_valid(output_address)) {
        fprintf(stderr,"[TB] Invalid configured SRAM address.\n");return 1;
    }

    /* Direct host preload, not a modeled SRAM transaction or an extra port.
     * The vector is stored in all four lanes so that I[r][c] = x[c] for every row. */
    system.sram.lines[input_address]=line_broadcast(pack_i8(input));
    printf("[TB] SRAM preload: line_address=0x%03" PRIX32 " word=0x%08" PRIX32 " (x4 lanes) input=",
           input_address,system.sram.lines[input_address].lane[0]);
    tb_print_vector(input);putchar('\n');

    uint64_t first_cycle=system.network.cycle;
    if(npu_run(&system,NPU_INFERENCE_CYCLES)!=NPU_RUN_COMPLETE) {
        fprintf(stderr,"[TB] Inference did not complete; no prediction reported.\n");
        return 1;
    }
    const Line128 *result=&system.sram.lines[output_address];
    for(unsigned lane=1;lane<4;lane++) if(result->lane[lane]!=result->lane[0]) {
        fprintf(stderr,"[TB] Result lanes differ; replicated store is inconsistent.\n");return 1;
    }
    uint32_t packed=result->lane[0];
    int8_t output[ARRAY_SIZE];
    for(unsigned i=0;i<ARRAY_SIZE;i++) output[i]=unpack_i8(packed,i);

    printf("\n========== Inference %" PRIu64 " : RESULT ==========\n",number);
    printf("Cycles: %" PRIu64 "\n",system.network.cycle-first_cycle);
    printf("Input:  ");tb_print_vector(input);putchar('\n');
    printf("Predicted output (quantized INT8): ");tb_print_vector(output);putchar('\n');
    printf("SRAM result: line_address=0x%03" PRIX32 " word=0x%08" PRIX32 " (x4 lanes)\n",
           output_address,packed);
    puts("================================================\n");
    return 0;
}

int tb_run(void)
{
    puts("Tiny NPU interactive testbench | 4 -> 4 -> 4");
    puts("Parameters: trained_npu_params.h; outputs are quantized INT8 values.");
    puts("Enter four decimal integers in [-128,127]. Each input runs a fresh inference.");
    puts("No simulated cycles advance while waiting. Enter q or send EOF to quit.\n");
    uint64_t number=1;
    for(;;) {
        int8_t input[ARRAY_SIZE];
        printf("[TB] Enter input [x1 x2 x3 x4], or q to quit: ");
        fflush(stdout); /* Prompt must be visible before the blocking read. */
        TbInputStatus status=tb_read_input(input);
        if(status==TB_QUIT) { puts("\n[TB] Finished.");return 0; }
        if(status==TB_IO_ERROR) { fprintf(stderr,"\n[TB] Terminal input failed.\n");return 1; }
        if(status==TB_INVALID) {
            puts("[TB] Invalid input: enter exactly four integers in [-128,127].\n");
            continue; /* No initialization, preload or cycles on invalid input. */
        }
        if(tb_infer(input,number)) return 1;
        number++;
    }
}
