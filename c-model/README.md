# Tiny NPU — SRAM → type-select demux → W / I / Bias registers → 4x4 MAC array

C99 cycle model of the updated draw.io architecture (page 01 of `Tiny-NPU.drawio`).
The 4→4→4 network and the INT8/INT32 arithmetic are unchanged; the datapath,
memory map and schedule are new. **An inference now takes 20 simulated cycles**
(was 28), with 8 wide reads, 2 wide writes and 32 useful MAC updates.

## Block map (diagram → source)

| Diagram block | File | Notes |
|---|---|---|
| NPU Controller / tile sequencer | controller.c/.h | Issues line address + 2-bit type; column-wave enables, POST, STORE |
| SRAM macro 1024 x 128-bit | sram.c/.h | 10-bit **line** address, 16 byte-write enables, registered Q |
| Type register + Data steering demux 1:4 | demux.c/.h | Type delayed one clock (= read latency) selects the destination |
| Weight / Input / Bias / Config registers | reg128.c/.h | One generic 128-bit register, instantiated four times |
| 4x4 MAC array (+ bias mux) | mac.c/.h | Cells hold only psum; W and I are byte lanes of their registers |
| Activation x4 (ReLU/bypass) | activation.c/.h | ReLU flag comes from the Config register |
| Requantize / clamp x4 | scale_clamp.c/.h | M and S come from the Config register |
| Result FIFO | fifo.c/.h | Holds q; write data is `{q,q,q,q}` |
| Wiring / run loop | npu.c/.h | Module order: controller, SRAM, demux, regs, array, activation, scale, FIFO |
| Bus framework | bus.c/.h | Unchanged |

`fetcher` is gone (as before). New: `demux.*`, `reg128.*`. Operand capture no
longer lives in mac.c or controller.c.

## Address map (10-bit line addresses, truth table from the diagram)

| Type select | Range | Used here |
|---|---|---|
| 00 Weights | 0x000–0x1FF | L1 W = 0x000, L2 W = 0x001 |
| 01 Inputs | 0x200–0x2FF | A = 0x200 (input / final output), B = 0x201 (hidden) |
| 10 Biases | 0x300–0x37F | L1 = 0x300, L2 = 0x301 |
| 11 Config | 0x380–0x38F | L1 = 0x380, L2 = 0x381 (range is my suggestion) |

The controller checks that every address lies in the range owned by its type,
and that a STORE target lies in the input range; a mismatch stops the run.
These are line indices, **not** the old byte addresses.

## Data layout

`Line128.lane[0]` is bits 31:0. Fixed wiring: `W[r][c]` / `I[r][c]` = byte `4r+c`
= lane r, byte c. Bias lane r = INT32 bias of row r. Config: lane0=M, lane1=S,
lane2=ReLU, lane3=0 (validated when the Config register captures).

**Input and result lines are stored replicated.** The array computes
`out[r] = bias[r] + Σc W[r][c]·I[r][c]`, so a matrix-vector product needs
`I[r][c] = x[c]` in every row: the 4-element vector is held in all four lanes.
The host preload (tb.c) does this, and STORE writes `{q,q,q,q}` with BE=0xFFFF so
the hidden vector can be read straight back as layer-2 input.

## Schedule (per layer, 10 edges; layer 2 = cycles 11..20)

| Edge | Action |
|---|---|
| E1 / E2 / E3 / E4 | Read W (00) / I (01) / B (10) / CFG (11); the tile start clears register-valid flags at E1 |
| E2 / E3 / E4 / E5 | Demux steers and registers capture W / I / B / CFG (one clock after each read) |
| E5..E8 | C1..C4: column c fires in all four rows; psum moves right one column per clock |
| E9 | POST: ReLU/bypass, requantize, FIFO push |
| E10 | STORE: 128-bit write, BE=0xFFFF, FIFO pop |

CFG is only needed at POST, so its capture overlaps C1. Column 1 takes its
psum_in from a 2:1 bias mux (`bias_enable` ? bias[r] : 0; always 1 for this network).
Registered semantics are kept: SRAM Q, register outputs and row sums are
registered links; control, demux outputs, activation and requantizer are wires.

## Build / test

```sh
make check                 # 5 notebook reference vectors, diffed against expected outputs
make run TB=1 LOG=flow     # interactive: four INT8 values per line, q to quit
make run TB=0 LOG=ports    # every block port and every register, each cycle
make log TB=0 LOG=flow     # save flow.log
```

`tests/vectors.expected` is generated from `npu_reference_vectors.json`
(Python integer model), not from this C model. `[1,2,3,4]` gives `[0,1,-4,-1]`,
packed `0xFFFC0100`.

## Limits

Numerical contract unchanged: INT8×INT8→INT16, INT32 psums and biases, overflow
is an error, `q = clamp_int8(round_ties_even(act·M / 2^S))`. This is an
architectural C model, not RTL. 20 cycles assumes a one-cycle SRAM response, one
request per clock, no stalls and a one-stage POST. Diagram pages 02–06 still show
the old 7-cycle diagonal schedule and byte-address map.
