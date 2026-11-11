#ifndef TB_H
#define TB_H
#include "stream.h"
/* 1=batch supplied, 0=EOF/quit, -1=invalid input. Values are already INT8. */
int tb_supply(Stream *,unsigned batch,unsigned features,bool interactive);
void tb_print_tensor(const char *,const Line128 *,size_t,unsigned batch,unsigned features);
#endif
