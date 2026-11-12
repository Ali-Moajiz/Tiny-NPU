/* Host-only loader regression helper. Never enters simulated execution. */
#include "sram.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv)
{
    assert(argc==3);
    Sram *s=calloc(1,sizeof *s);assert(s);
    memset(s->regs.memory,0xA5,sizeof s->regs.memory);
    s->regs.response.valid=true;s->regs.reads=7;s->regs.writes=9;
    char error[256];bool ok=sram_load_hex(s,argv[1],error,sizeof error);
    if(!strcmp(argv[2],"reject")) {
        assert(!ok);
        const unsigned char *bytes=(const unsigned char *)s->regs.memory;
        for(size_t i=0;i<sizeof s->regs.memory;i++) assert(bytes[i]==0xA5);
        assert(s->regs.response.valid&&s->regs.reads==7&&s->regs.writes==9);
        puts(error);
    } else {
        if(!ok) fprintf(stderr,"%s\n",error);
        assert(ok);
        FILE *f=fopen(argv[2],"rb");assert(f);
        for(unsigned i=0;i<SRAM_LINES;i++) for(unsigned b=0;b<16;b++) {
            int expected=fgetc(f);assert(expected!=EOF);
            assert((uint8_t)line_byte(s->regs.memory[i],b)==(unsigned)expected);
        }
        assert(fgetc(f)==EOF&&!ferror(f));assert(!fclose(f));
        assert(!s->regs.response.valid&&!s->regs.reads&&!s->regs.writes);
    }
    free(s);return 0;
}
