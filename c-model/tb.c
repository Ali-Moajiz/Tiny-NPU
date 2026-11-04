#include "tb.h"
#include "npu.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void tb_print_vector(const int8_t values[ARRAY_SIZE])
{
    printf("[%d, %d, %d, %d]",values[0],values[1],values[2],values[3]);
}

static int tb_infer(const int8_t input[4][4],uint64_t number)
{
    Npu system;
    printf("\n========== Inference batch %" PRIu64 " : START ==========\n",number);
    if(!npu_init(&system)) { fprintf(stderr,"[TB] NPU initialization failed.\n");return 1; }
    unsigned batch=system.controller.options.batch;
    uint32_t input_address=system.controller.layers[0].input_base;
    uint32_t output_address=system.controller.layers[NETWORK_LAYERS-1].output_base;
    /* One independent vector per lane. No replication to fake utilization.
     * Direct host preload occurs while idle, outside the simulated port timing. */
    for(unsigned v=0;v<batch;v++) system.data.sram.regs.lines[input_address].lane[v]=pack_i8(input[v]);
    printf("[TB] SRAM preload: line_address=0x%03" PRIX32 " independent_vectors=%u\n",input_address,batch);
    uint64_t first_cycle=system.cycle;
    if(npu_run(&system,NPU_INFERENCE_CYCLES)!=NPU_RUN_COMPLETE) {
        fprintf(stderr,"[TB] Inference did not complete; no prediction reported.\n");return 1;
    }
    const Line128 *result=&system.data.sram.regs.lines[output_address];
    printf("\n========== Inference batch %" PRIu64 " : RESULT ==========\n",number);
    printf("Cycles for complete batch: %" PRIu64 "\n",system.cycle-first_cycle);
    for(unsigned v=0;v<batch;v++) {
        int8_t output[4];
        for(unsigned r=0;r<4;r++) output[r]=unpack_i8(result->lane[v],r);
        printf("Vector %u input: ",v+1);tb_print_vector(input[v]);putchar('\n');
        printf("Predicted output (quantized INT8): ");tb_print_vector(output);putchar('\n');
        printf("SRAM result: line_address=0x%03" PRIX32 " lane=%u word=0x%08" PRIX32 "\n",output_address,v,result->lane[v]);
    }
    printf("Useful MACs: %u; compute clocks: %u; peak active: %u/16\n",system.data.array.regs.updates,system.data.array.regs.compute_cycles,system.data.array.regs.peak_active);
    printf("Compute utilization: %.2f%%; whole-run utilization: %.2f%%\n",
           100.0*system.data.array.regs.updates/(16*system.data.array.regs.compute_cycles),
           100.0*system.data.array.regs.updates/(16*(system.cycle-first_cycle)));
    puts("================================================\n");return 0;
}
int tb_run(void)
{
    const unsigned total_batches = 100;

    srand(42);
    for (unsigned number = 1; number <= total_batches; number++) {
        int8_t input[4][4] = {{0}};
        for (unsigned v = 0; v < NPU_BATCH; v++) {
            for (unsigned k = 0; k < ARRAY_SIZE; k++)
                input[v][k] = (int8_t)((rand() % 256) - 128);
        }
        if (tb_infer(input, number))
            return 1;
    }
    return 0;
}
