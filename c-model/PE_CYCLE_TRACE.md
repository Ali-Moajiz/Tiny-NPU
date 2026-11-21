# PE cycle contract and reference trace

This trace uses the nine-instruction, batch-4, N=4, K=4 program in
`tests/check_model.py::golden_pipeline`. Clock numbers start at 1. Values
captured at the end of clock C are available to consumers in clock C+1.

| Clocks | Instruction / operation | Datapath effect |
|---|---|---|
| 1–5 | STREAM_LOAD | Input beat writes SRAM at clock 4 |
| 6–11 | LOAD_OPERAND B | Bias slot capture at 10 |
| 12–17 | LOAD_OPERAND CFG | Config slot capture at 16 |
| 18–23 | LOAD_OPERAND W | Weight slot capture at 22 |
| 24–29 | LOAD_OPERAND X | Input slot capture at 28 |
| 30–32 | EXEC_MAC fetch / response / decode | Operands held |
| 33 | EXEC_MAC k=0 | 16 accumulator captures: bias + x*w |
| 34 | EXEC_MAC k=1 | 16 captures: prior sum + x*w |
| 35 | EXEC_MAC k=2 | 16 captures: prior sum + x*w |
| 36 | EXEC_MAC k=3 | 16 captures; tile complete |
| 37 | EXEC_MAC update | Advance PC |
| 38–40 | POST_PROC fetch / response / decode | Accumulators held |
| 41 | POST column 0 | ACT captures column 0; packer clears output tile |
| 42 | POST column 1 | ACT column 1; SCALE column 0 |
| 43 | POST column 2 | ACT column 2; SCALE column 1; PACK column 0 |
| 44 | POST column 3 | ACT column 3; SCALE column 2; PACK column 1 |
| 45 | Pipeline drain | SCALE column 3; PACK column 2 |
| 46 | Pipeline drain | PACK column 3; result register becomes complete |
| 47 | POST store | Registered result writes SRAM |
| 48 | POST update | Advance PC |
| 49–54 | STREAM_STORE | Read at 52; output accepted at 53 |
| 55–59 | HALT lifecycle | Done at update, clock 59 |

Each ACT/SCALE column has up to four independent vector lanes. MAC compute
uses 16 cells simultaneously; a column is selected only during readout.
Processing columns sequentially in POST does not serialize MAC k-steps.
For N<4 only N columns are read. For batch<4 the inactive lanes carry invalid
bubbles, and packed output padding remains zero.

Compute work: 4 clocks × 16 active cells = 64 useful MAC updates.
Compute utilization: 100%. Whole-run utilization: 64 / (59 × 16) = 6.78%.
Instruction fetch/decode overhead dominates this small program; modular PE
instances do not remove that overhead or introduce instruction overlap.

## Register and wire ownership

| Instance | Registers owned | Inputs | Output |
|---|---|---|---|
| `mac` | Two X/W/B/CFG operand slots; compute/readout metadata | SRAM response, forward result, control | Shared X/W/bias wires; selected accumulator columns |
| `mac.cell[v][n]` | INT32 accumulator, valid | `x[v]`, `w[n]`, `bias[n]`, control | Private `sum[v][n]` |
| `activation.lane[v]` | INT32 activated result, valid, context | Selected accumulator lane | Registered activation lane |
| `scale.lane[v]` | INT8 scaled result, valid, context | Activation lane | Registered byte lane |
| `scale` packer | 128-bit result, valid, next column/context | Four scaled byte lanes, POST start | Held result for SRAM / forward |

The bus framework uses ordinary typed point-to-point/shared wires. It is not
AXI and adds no request/response latency itself. SRAM latency and register
boundaries account for data availability. All blocks publish old registered
outputs, evaluate combinational logic, validate, then capture on one shared
edge. A failed validation prevents every block's clock callback.

## Numerical compatibility and limits

Bias is added once on k=0 of a non-retaining MAC instruction. Retaining
instructions continue existing sums across operand reloads. POST applies
activation, then signed 64-bit multiply, nearest-even shift and INT8 clamp.
Overflow in INT32 accumulation is a model fault, not saturation or wrapping.
Forwarding copies the packed result without transposing vector/neuron order.

The four-lane pipeline replaces the old whole-tile post operation. For N=4,
POST dispatch costs 7 clocks with store (previously 2), or 6 without store
(previously 2 at POST_LATENCY=1). Existing binaries retain their arithmetic and
addresses, but cycle predictions must use N+2+store. An 8x8 array and tanh or
sigmoid approximations are separate architectural changes, not implicit here.
