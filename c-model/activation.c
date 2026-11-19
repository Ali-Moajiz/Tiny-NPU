#include "activation.h"
void activation_init(Activation *a)
{
    for (unsigned i=0; i<ARRAY_SIZE; i++) {
        a->lane[i].lane = i;
        a->lane[i].ports.in.value = a->ports.in.lane[i];
        a->lane[i].ports.out.value = a->ports.out.lane[i];
    }
}
