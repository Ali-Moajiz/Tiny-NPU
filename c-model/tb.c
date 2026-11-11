#include "tb.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
