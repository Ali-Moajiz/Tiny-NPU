#ifndef SCALE_CLAMP_H
#define SCALE_CLAMP_H
#include "datapath.h"
typedef struct {
    Bus output_bus;
    struct { const Bus *control,*activated,*config;Bus *quantized; } ports;
    bool error;
} ScaleClamp;
/* Requires multiplier in [1,INT32_MAX], shift in [0,62]. M = Config lane0, S = Config lane1;
 * the Config register validates the fields when it captures them. */
int8_t scale_clamp_value(int32_t value,int32_t multiplier,unsigned shift);
bool scale_clamp_init(ScaleClamp *,BusNetwork *,const Bus *control,const Bus *activated,const Bus *config);
void scaleClampBusses(void *,uint64_t);
#endif
