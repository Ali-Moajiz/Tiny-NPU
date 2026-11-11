#include "isa.h"
Instruction instruction_decode(Line128 r)
{
    Instruction i;
    i.opcode=r.lane[3]>>28;
    i.src=(r.lane[2]>>28)|((r.lane[3]&0x0fffffffU)<<4);
    i.dest=(r.lane[1]>>28)|((r.lane[2]&0x0fffffffU)<<4);
    i.dims=r.lane[1]&0x0fffffffU;
    i.flags=r.lane[0];
    return i;
}
bool isa_address(uint32_t address, uint32_t start, uint32_t end)
{ return !(address&15u) && address>=start && address<end; }
