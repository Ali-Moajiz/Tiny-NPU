#ifndef PE_MATH_H
#define PE_MATH_H
#include <stdbool.h>
#include <stdint.h>

/* Datapath operation: signed magnitude, quotient/remainder, tie-to-even mux.
 * Callers validate shift <= 62. No floating point or signed right shift. */
static inline int64_t pe_round_shift(int64_t value, unsigned shift)
{
    if (!shift) return value;
    bool negative = value < 0;
    uint64_t magnitude = negative ? (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;
    uint64_t divisor = UINT64_C(1) << shift;
    uint64_t quotient = magnitude / divisor;
    uint64_t remainder = magnitude % divisor;
    if (remainder > divisor / 2 || (remainder == divisor / 2 && (quotient & 1)))
        quotient++;
    return negative ? -(int64_t)quotient : (int64_t)quotient;
}
#endif
