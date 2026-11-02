# Tiny NPU — ping-pong parameters, outer-product array, quantized layer bypass

Updated from the attached `tiny-npu.zip`, retaining the block modules, generic bus scheduler,
128-bit native SRAM port, trained parameters and optional terminal TB. No extra block C/H
files are required. This is an architectural C99 simulator, **not a synthesizable HLS top**.
The modeled clocks are explicit scheduling assumptions, not measured RTL timing.

The important change is the **meaning of a PE**: `PE[r][v]` now accumulates output neuron r
for independent input vector v. The input/reduction index k advances across four clocks.
The old `PE[r][k]` column-by-column/psum-right schedule has been replaced. This is an
**outer-product engine with local accumulators**, not a diagonal systolic implementation.

## Run it

```sh
make run TB=1 LOG=flow                  # one vector per inference, existing interaction
make run TB=1 BATCH=4 LOG=flow          # collect FOUR independent vectors, then run a batch
make run TB=0 BATCH=4 LOG=ports         # built-in distinct inputs, every block/port each edge
make run BATCH=4 PREFETCH=0 FUSE=0      # disable both optimizations to compare scheduling
make check                             # independent numerical + timing/masking/fault regressions
```

`LOG=flow|ports|both|none`; `TB=0|1`; `BATCH=1|2|3|4`; `PREFETCH=0|1`; `FUSE=0|1`.
Every option combination has its own build directory; no clean is needed when changing flags.
TB sources are absent from TB=0 builds. Neither build needs HLS tools or an HLS library.
Inputs are already-quantized INT8 values, not original floating features. Host preprocessing
metadata remains in `npu_preprocessing.json`. The terminal TB waits without advancing clocks,
rejects malformed lines, and discards an incomplete batch on q/EOF. It resets the NPU and
preloads SRAM between batches; parameter caching across separate TB runs is not claimed.

For `BATCH=4`, enter four lines such as:

```text
1 2 3 4
-1 0 1 2
10 -20 30 -40
127 -128 64 -64
```

Each vector gets its own prediction. `[1,2,3,4]` still produces `[0,1,-4,-1]`.

## Architecture and source map

| Block | Source | Behavior |
|---|---|---|
| Controller / AGU | controller.c/.h | FSM, loaded-flag checks, background parameter reads, bank selection, final store |
| SRAM / wrapper | sram.c/.h | 1024x128, one read OR write per clock, one-cycle read response, 16 byte enables |
| Steering demux | demux.c/.h | Delay TYPE **and destination BANK** with the read; route previous response |
| W/I/B/CFG register files | reg128.c/.h | Two 128-bit banks per object; hold unselected bank; reject active-bank overwrite |
| 4x4 MAC engine | mac.c/.h | Row-weight and vector-column-input broadcasts; 16 local INT32 accumulators |
| Activation | activation.c/.h | Registered INT32 tile; configured ReLU or bypass |
| Requantizer | scale_clamp.c/.h | Registered INT8 tile, model-specific M/S, nearest-even rounding, saturation |
| Result FIFO | fifo.c/.h | One 128-bit output tile retained until SRAM accepts STORE |
| Integration | npu.c/.h | Connect the buses, register callbacks, initialize/run/check faults |
| Generic framework | bus.c/.h | **Unchanged** callback traversal and registered/wire semantics |

The supplied draw.io XML shows the current top-level and PE mapping. Older internal sheets
are explicitly historical references. `CYCLE_TRACE.md` explains the actual 20-edge schedule.

### Outer-product math

For reduction step k and every active PE:

```text
weight[r] = W[r,k]                     # 4 selected INT8 weights; broadcast along rows
input[v]  = X[v,k]                     # up to 4 selected INT8 inputs; broadcast down columns
acc[r,v]  = (k==0 ? bias[r] : old_acc[r,v]) + weight[r] * input[v]
```

This computes `Y[r,v] = bias[r] + sum_k W[r,k]*X[v,k]`, equivalent to `X @ W.T` for
batch-major Python arrays. Each weight is reused across up to four distinct vectors; each
input is reused across four output neurons. A cell uses its OWN previous accumulator.
The simulator evaluates every next sum before committing PE state. There is no horizontal
combinational adder chain and no neighbor-to-neighbor input propagation in this mapping.
Bias is added exactly once, at k=0. Masked vector columns do no useful MAC updates.

A full batch of four has 16 useful MAC updates per compute clock; one vector has four.
Broadcasting duplicate inputs into four lanes is **not** used to inflate utilization.
Four parallel weight-byte muxes and four input-byte muxes select k from the local banks.
The SRAM does not need to supply a new input or weight for each MAC update.

### Ping-pong and prefetch

Bank0 contains layer1 W/B/CFG; bank1 contains layer2 W/B/CFG. During layer1 compute clocks
C1/C2/C3, the AGU issues layer2 W/B/CFG reads through the same native SRAM port. The delayed
bank tag sends each response to the shadow bank while active bank0 is held stable.
The controller selects bank1 only after the new input AND every parameter bank is loaded.
This is local SRAM-to-register prefetch, **not an external DMA engine**, a CPU, or an IREE
compiler. The two layer descriptors currently come from npu.c. More layers/tiles would need
bank-generation/ownership management before reusing a physical bank.

### Hidden tile bypass and registered stages

With `FUSE=1`, the layer1 accumulator tile passes through ReLU, then model-specific INT8
requantization. The registered INT8 tile feeds the inactive input bank through a 2:1 input
mux: SRAM response or quantized hidden result. It does **not** forward raw INT32 sums to an
INT8 layer. Hidden SRAM at0x201 is neither written nor read. Only final results enter FIFO.
With `FUSE=0`, the existing SRAM write/read handoff remains available as a reference path.
This implements the known two-layer dense/ReLU chain, not general compiler operator fusion.

SRAM Q, register-bank Q, final MAC tile, activation tile, requantized tile and FIFO outputs
are registered links. Control and demux routing are combinational wires. Thus a producer
running earlier in the callback list cannot expose its newly captured register to a consumer
at that same edge. No extra stages are hidden inside a tick counter. A 4x4 requantization
stage assumes up to16 parallel scale paths; a smaller physical postprocessor would require
serialization and different cycle counts. SRAM and scale-stage timing have not been synthesized.

## Memory / register format

| TYPE | SRAM line range | Example objects |
|---|---|---|
| 00 W | 0x000–0x1FF | W1=000, W2=001 |
| 01 I | 0x200–0x2FF | X/final Y=200; unfused hidden=201 |
| 10 B | 0x300–0x37F | B1=300, B2=301 |
| 11 CFG | 0x380–0x38F | CFG1=380, CFG2=381 |

Line addresses are 10-bit indices, equivalent byte address=`line<<4`. Total SRAM=16KiB.
`Line128.lane[0]` is bits31:0, independent of host endian.

- W lane r contains four signed bytes W[r,0..3].
- I lane v contains four signed input features for independent vector v.
- B lane r contains one signed INT32 accumulator-domain bias.
- CFG: lane0 positive INT32 M, lane1 shift S0..62, lane2 ReLU0/1, lane3 reserved0.
- Quantized output lane v contains four output neurons for vector v. No replicated lanes.
- STORE byte enables: batch1=000F, batch2=00FF, batch3=0FFF, batch4=FFFF. Inactive bytes hold.

Quantization contract is unchanged: signed INT8 multiply, INT32 accumulation with overflow
reported as an error, then `clamp_int8(round_nearest_even((int64)activated * M / 2^S))`.
No independent bias scales, dynamic per-input output scale, or float MAC arithmetic.

## Measured simulator results

For the supplied two-layer network, independent of valid batch width1..4:

| PREFETCH | FUSE | Batch cycles | SRAM reads | SRAM writes |
|---:|---:|---:|---:|---:|
| 0 | 0 | 26 | 8 | 2 |
| 0 | 1 | 24 | 7 | 1 |
| 1 | 0 | 23 | 8 | 2 |
| 1 | 1 | **20** | **7** | **1** |

Each batch needs eight compute clocks (four per layer). Batch4 executes128 useful MAC
updates: 100% utilization **during compute**, 40% over the complete20-cycle run.
Batch1 executes32 updates:25% during compute,10% overall. Single-vector latency remains20
cycles versus the uploaded model; no single-vector speedup is claimed. The array now processes
four distinct vectors in those20 cycles rather than four serial20-cycle inferences.
This is batch throughput; batch collection time, host preload, clock frequency and RTL area
are excluded. The stages differ from the uploaded single-cycle POST, so compare actual work
and full measured schedules rather than assuming every stage is unchanged.

`make check` checks1044 batch inferences (2580 independent vectors), every intermediate MAC
accumulator against a Python integer oracle, all four feature combinations, partial batches,
untouched SRAM byte lanes, active-bank isolation, registered forwarding and timing statistics.
It also tests invalid config/address, overflow, illegal bank overwrite and FIFO-underflow
faults, and nearest-even boundaries for all shifts0..62. Expected fault diagnostics appear
on stderr. The temporary check library uses UBSan. Compiler support for UBSan is required
for this check target, not for running the model.

## What this borrows from Coral

The first-generation Edge TPU and the open RISC-V Coral NPU are different designs.
Google's current matrix-engine documentation describes an **outer-product** formulation,
vector-register source operands and local matrix accumulators; it is not evidence that this
particular16-PE weight-stationary systolic grid is Coral's implementation. The published256
MACs/clock is Coral's configured engine width, not a number this4x4 model can achieve.
The official compiler description includes fusion and tiling; that does not imply every
intermediate tensor of every network bypasses SRAM. Here, bypass is legal because the entire
known hidden tile fits in local registers and matches layer2's quantization contract.

Primary references:
- https://developers.google.com/coral/guides/hardware/vme-matrix-engine
- https://developers.google.com/coral/guides/hardware/datasheet
- https://developers.googleblog.com/unifying-edge-ai-how-coral-npus-open-standard-brings-portable-energy-viable-intelligence-from-model-to-silicon/

No RISC-V core, external DMA, compiler, variable-latency/backpressure protocol, automatic
large-matrix tiler or HLS-synthesizable top has been added. These remain future work.
