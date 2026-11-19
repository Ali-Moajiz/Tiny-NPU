# Tiny NPU C model

Execute the Tiny NPU compiler's `sram.hex` image with cycle-by-cycle logging.
Requires a C99 compiler, Make, and Python 3 for tests. No HLS installation is required.

## Run

From `Tiny-NPU/c-model/`, run an image compiled with `--debug` (which emits
`sram.hex`). Set `BATCH` to the batch size used by the compiler.

```sh
# Random quantized inputs; example: a 4->8->8->2 model, batch 4.
make -s run TB=1 BATCH=4 LOG=flow SRAM=../output/sram.hex \
  ARGS="--runs 10 --input-features 4 --output-features 2" 2>&1 | tee npu.log

# Enter one batch, view its result, then enter another. Type q to quit.
make -s run TB=1 BATCH=4 LOG=flow SRAM=../output/sram.hex \
  ARGS="--interactive --input-features 4 --output-features 2"

# Run without the testbench, using packed input/output bytes.
make -s run TB=0 BATCH=4 LOG=flow SRAM=../output/sram.hex \
  ARGS="--input-bin input.bin --output-bin output.bin"
```

TB defaults: 100 batches, seed 42, four input and four output features. Use
`--seed N` to change the random sequence and `--runs N` to change the run count.
Each inference resets execution and restores the original preloaded memory
image. The file is read once per process, not once per cycle or inference.
`tb.c` is excluded when `TB=0`.

Inputs are **already quantized signed INT8**, not floating-point features.
Apply the compiler's exported input scale before supplying real-world values.
`--input-features` and `--output-features` describe host packing/display; the
instruction words determine compute dimensions. `BATCH` is checked against
those words and does not recompile the network.

Packed binary input uses one 16-byte beat per four feature channels: byte
`4*vector + channel_in_tile`. Zero-pad unused vectors/channels. For more than
four channels, send consecutive channel tiles. Output uses the same layout.
`--input-bin` executes once and overrides random input generation.

Build options: `LOG=flow|ports|both|none` and
`SANITIZE=1`. Binaries are placed in separate `build/` directories for each
configuration. `--max-cycles N` limits execution (default 100000).

## Model organization

| Files | Responsibility |
|---|---|
| `main.c`, `tb.c` | Host file/terminal I/O and optional random testbench |
| `npu_top.c` | Top-level instances, port connections, registered block order |
| `bus.c`, `framework.c` | Checked point-to-point buses and cycle phases |
| `isa.c` | Decode the 128-bit instruction fields |
| `controller.c` | Fetch, response, decode, dispatch, update lifecycle |
| `instructions.c` | Dedicated instruction callbacks and opcode registration |
| `sram.c` | Hex preload, single-port 65536x128 memory, registered read response |
| `stream.c` | Simulation input/output stream endpoint and back-pressure |
| `mac.c` | Two operand slots, row/column broadcast muxes, accumulator readout |
| `mac_pe.c` | One INT8 multiplier, bias/feedback mux, checked adder and private INT32 accumulator |
| `activation.c`, `activation_pe.c` | Four registered activation PE instances |
| `scale_clamp.c`, `scale_pe.c` | Four registered scaling PEs and a result packer |
| `pe_signals.h`, `pe_math.h` | Typed PE wires and integer rounding primitive |

Each stateful block owns `regs` and `ports`. Only `npu_top.c` connects the port
pointers. There are no `q/d` structures. The generic framework calls every
registered block through **publish -> evaluate -> check -> clock** phases.
Registers change only in the clock phase, after every transfer has passed
validation. A block never observes another block's new register value on the
same edge. Combinational outputs can propagate during evaluation; block order
must follow those combinational dependencies.

The SRAM response is a broadcast 128-bit data bus. Controller latch controls
select slot 0/1 and X/W/B/CFG inside the MAC. The controller receives instruction
words from SRAM, but operand/result data goes directly between datapath blocks.
Results can write SRAM or forward to another MAC input slot. The runtime stream
endpoint is a simulator peripheral, not an internal post-processing FIFO.

## PE instances and timing

The active array is still **4x4 broadcast**, with 16 `MacPe` instances at
`n.mac.cell[vector][neuron]`. All cells in one row share `x[vector]`; all cells
in one column share `w[neuron]` and `bias[neuron]`. Each cell has its own
accumulator output bus. There is no PE-to-PE forwarding and no adder tree.

At compute step k, every enabled cell calculates:

`acc[vector][neuron] = (bias if k=0 and !accumulate else acc) + X[vector][k] * W[neuron][k]`

The signed 8x8 product is 16 bits. The feedback adder checks the INT32 range
before capturing the accumulator; overflow faults the entire edge. Disabled
cells hold their registers. X/W operand registers remain in the MAC's two
shared slots, avoiding a redundant register stage in every cell. `regs` holds
state, `ports` holds connections, and `comb` names combinational wires. A product
in `comb` is not a pipeline register. One k-step still takes one compute clock.

Four `ActivationPe` instances and four `ScalePe` instances process one neuron
column across the batch each clock. Each PE owns its registered output and
valid bit. The scale PE has a signed 32x32->64 product, nearest-even rounded
shift and INT8 saturation. The packer preserves byte `4*vector + neuron` and
zero-fills inactive bytes. Results remain available for SRAM write or FORWARD
until the next POST starts. No internal result FIFO is added.

For N=4, relative to the first POST dispatch clock:

| Clock | Activation captures | Scaling captures | Packer captures | SRAM |
|---|---|---|---|---|
| T | Column 0 | — | Reset output tile | — |
| T+1 | Column 1 | Column 0 | — | — |
| T+2 | Column 2 | Column 1 | Column 0 | — |
| T+3 | Column 3 | Column 2 | Column 1 | — |
| T+4 | — | Column 3 | Column 2 | — |
| T+5 | — | — | Column 3; result complete | — |
| T+6 | — | — | Hold | Optional result store |

POST dispatch therefore takes `N+2` clocks without a store, or `N+3` with a
store. Both PE stages always have one register boundary; the old
`POST_LATENCY=1` is accepted only for compatibility with existing scripts and
emits a deprecation warning; `POST_LATENCY=0` is rejected. Instruction encoding and numerical
results for existing bypass/ReLU programs stay compatible. See
[PE_CYCLE_TRACE.md](PE_CYCLE_TRACE.md) for a full-program schedule.

CFG is four little-endian 32-bit words: positive INT32 multiplier M, shift S
(0..62), activation mode, and activation parameter. Activation acts in
**accumulator units before requantization**, not on a dequantized float.

| Mode | Activation | Parameter (CFG word 3) | Integer behavior |
|---|---|---|---|
| 0 | Bypass | 0 | x |
| 1 | ReLU | 0 | max(0, x) |
| 2 | LeakyReLU | p = 0..31 | x if nonnegative; otherwise round-even(x / 2^p) |
| 3 | Clipped ReLU | positive INT32 ceiling C | min(C, max(0, x)) |

Modes 2/3 extend the former reserved configuration fields. The existing
compiler still emits only modes 0/1; compiler support is a separate change.
A ceiling of 6 means six accumulator units, **not automatically ReLU6** in
real units. Tanh/sigmoid are not implemented: they need a specified input scale,
fixed-point output format and approximation error budget. Unsupported modes
are rejected rather than silently approximated with host floating-point math.

`PostTag` carries configuration and column/batch context through the pipeline
so each value can be checked before packing. PC is diagnostic metadata; this
C transport layout is not a requirement to synthesize a PC register per lane.
The fixed compiler ABI and SRAM beat packing still target 4x4, batch 1..4.
Changing `ARRAY_SIZE` alone does not implement 8x8 support.

## SRAM and instruction ABI

`sram.hex` must contain exactly 65,536 data words of 32 hex digits. Each line is bits
127..0, with the lowest-address byte at the right. Blank lines and `#`/`//`
comments are accepted. Malformed or incomplete files are rejected before
execution. SRAM accesses use byte addresses, aligned to 16 bytes.

| Byte address range | Contents |
|---|---|
| `0x00000–0xBFFFF` | INT8 weights (768 KiB) |
| `0xC0000–0xDFFFF` | Input/intermediate/output tiles |
| `0xE0000–0xEFFFF` | INT32 biases in accumulator units |
| `0xF0000–0xF0FFF` | CFG: multiplier, shift, activation mode, activation parameter |
| `0xF1000–0xFFFFF` | Instructions; reset PC = `0xF1000` |

The SRAM is one 1 MiB block: 65,536 × 128 bits. Region ends are exclusive
`0xC0000`, `0xE0000`, `0xF0000`, `0xF1000`, `0x100000`; capacities are 768,
128, 64, 4 and 60 KiB. Physical word address is byte address / 16 (16 bits).
The final beat is `0xFFFF0`; `0x100000` is outside memory. The loader stages
its image on the host heap and commits only a fully validated image; the NPU
and reset snapshot have static host storage to avoid large stack allocations.
File I/O occurs only at initialization, never during simulated memory accesses.

Compiler and model must use the same memory map. **Recompile old 16 KiB images**:

```sh
# From the repository root; alternatively use python3 compiler/main.py.
./compiler/bin/tiny-npu-compiler model.onnx --calibration samples.npy --out output --debug
make -C c-model run TB=1 BATCH=4 LOG=flow SRAM=../output/sram.hex \
  ARGS="--runs 100 --seed 42 --input-features 4 --output-features 2"
```

To rebuild the executable, install compiler/requirements.txt and
`pyinstaller==6.16.0 pyinstaller-hooks-contrib==2025.9`, then run
`make -C compiler binary PYTHON=python3`. `sram.bin` is exactly 1,048,576 bytes;
`params.bin` is 987,136 bytes including unused gaps before entry PC.
`program.bin`/`program.hex` contain only instructions, not a full SRAM image.
Hex and other debug reports still require `--debug`.

The program region holds 3,840 instructions. Stream buffers still hold 256
beats (up to 1,024 input/output features), and batches remain 1–4. Only existing
operators, two operand slots and the existing forwarding cases are supported.
Increasing SRAM does not change instruction encoding or dispatch timing.

Instruction layout: opcode `[127:124]`, src `[123:92]`, dest `[91:60]`, dims
`[59:32]`, flags `[31:0]`. Dims packs N in `[3:0]`, batch in `[7:4]`, K in
`[11:8]`. LOAD_OPERAND destination is `(slot << 2) | field`, with X=0, W=1,
B=2, CFG=3. Every load/store transfers exactly one 128-bit beat.

| Opcode | Instruction | Dispatch cycles, without back-pressure |
|---|---|---|
| `1` | STREAM_LOAD | 1 |
| `2` | EXEC_MAC | K |
| `3` | POST_PROC | N + 2 + store (store is 0 or 1) |
| `4` | FORWARD | 1 |
| `5` | STREAM_STORE | 2 |
| `6`, `7` | LOOP_START, LOOP_END | 1 each |
| `8` | LOAD_OPERAND | 2 |
| `F` | HALT | 1 |

Each instruction also costs four lifecycle cycles: fetch request, fetch
response, decode, update. SRAM reads complete one cycle after request; their
response remains valid until consumed. Stream transfers wait for availability.
Only one non-nested hardware loop is supported, matching the compiler ABI.

EXEC_MAC flags: bit 0 selects the slot, bit 1 retains accumulators. Otherwise,
MAC k=0 seeds each accumulator with its neuron bias. One clock updates N*batch
PEs. Partial sums survive intervening instructions and operand loads. POST
consumes the completed accumulator tile column by column; activation precedes INT64 multiply,
nearest-even shift and signed INT8 clamp. Inactive output bytes are zero.

This is a sequential instruction engine, not an overlapping issue pipeline.
Two operand slots do not automatically overlap loads with compute. Prefetch and
forwarding depend on the emitted program; obsolete `PREFETCH`/`FUSE` build flags
are rejected. For a 42-instruction 4->8->8->2 batch-4 program, the measured totals
are 289 cycles, with 32 compute clocks and 448 useful MAC updates. The previous
whole-tile post implementation took 266 cycles with one output register. Four
activation/scaling lanes use fewer arithmetic units but require more clocks;
this refactor does not claim a throughput improvement. Different instruction
programs have different cycle counts.

Flow logs identify PC, opcode, k, active PEs, input, weight, product, prior sum,
and updated accumulator. ACT, SCALE, PACK and READOUT events include the column
and lane; `available_cycle` states when a registered output can be consumed. Metrics include instructions, useful MACs, SRAM
reads/writes, compute utilization and whole-run utilization. SRAM reads include
instruction fetches. `bus_stall_cycles` counts extra waits for unavailable
responses/input/output readiness; it excludes normal fetch and response cycles.

## Add an instruction or block

For an instruction, implement its validator and execution callback in
`instructions.c`, then add an `InstructionHandler` descriptor to `builtins[]`.
Alternatively register a static descriptor with `controller_register()` after
initialization. Duplicate opcodes, null callbacks and out-of-range opcodes are
rejected. Descriptors must remain alive during execution. `tests/check_core.c`
shows a custom opcode registered without changing the controller FSM.

An execution callback emits control signals and returns DONE, PROGRESS, WAIT,
or FAULT. PROGRESS advances the instruction's microstep; WAIT holds it. An
optional retirement callback updates controller state at the update edge. Do
not mutate datapath registers from an instruction callback.

For a block, define its registers/ports in its own header, supply `BlockOps`,
connect its typed buses in `npu_top.c`, and call `framework_register()` once
during initialization. The framework visits the registered instances each cycle.
All buses have one designated driver; consumers share that bus instance.

The callback registry, file loader, logs and host runner are simulation software.
They are not directly synthesizable HLS RTL; a hardware backend needs static
opcode decoding and explicit hardware interfaces with the same timing contract.

## Test

```sh
make -s test LOG=none
make -s test LOG=both SANITIZE=1

# Optional: compare with the existing compiler's emitted binaries and oracle.
make -s LOG=flow
python3 tests/check_model.py \
  --binary build/tb0-b1-pe-flow-san0/tiny_npu --compiler ../compiler

# Independent random/interactive TB checks (requires compiler dependencies).
make -s TB=1 BATCH=4 LOG=flow
python3 tests/check_tb.py --binary build/tb1-b4-pe-flow-san0/tiny_npu --compiler ../compiler
```

The built-in suite uses only Python's standard library. The optional compiler
comparison needs the compiler's Python dependencies and `LOG=flow` or `both`.
PE tests additionally check shared-wire identity, private register state, disabled
cell holds, bias vs. feedback selection, no same-edge leakage, stage-order
independence across register boundaries, malformed tags, and atomic overflow
rejection. A hand-checked trace tests the four-lane POST schedule.
Tests cover all implemented opcodes, partial tiles, batches 1–4, retained sums,
forwarding, back-pressure, signed rounding, masked stores, malformed images,
overflow, and invalid instruction/state rejection. Expected fault tests print
error messages. Map tests check every region's first/last beat, exclusive end,
alignment, and access to the final physical beat. The image loader helper
checks every loaded byte and transactional rejection of malformed files.
The optional compiler comparison includes a 4->100->100->1 network whose
weights exceed the old 8 KiB region. On hosts that prohibit LeakSanitizer tracing, use
`ASAN_OPTIONS=detect_leaks=0`; address and undefined-behavior checks remain enabled.
