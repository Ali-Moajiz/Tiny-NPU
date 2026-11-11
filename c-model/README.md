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

Build options: `LOG=flow|ports|both|none`, `POST_LATENCY=0|1` (default 1), and
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
| `sram.c` | Hex preload, single-port 1024x128 memory, registered read response |
| `stream.c` | Simulation input/output stream endpoint and back-pressure |
| `mac.c` | Two operand slots inside the MAC and the 4x4 accumulator array |
| `activation.c` | Combinational ReLU/bypass |
| `scale_clamp.c` | Nearest-even scaling, INT8 clamping, optional output latency |

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

## SRAM and instruction ABI

`sram.hex` must contain exactly 1024 lines of 32 hex digits. Each line is bits
127..0, with the lowest-address byte at the right. Blank lines and `#`/`//`
comments are accepted. Malformed or incomplete files are rejected before
execution. SRAM accesses use byte addresses, aligned to 16 bytes.

| Byte address range | Contents |
|---|---|
| `0x0000–0x1FFF` | INT8 weights |
| `0x2000–0x2FFF` | Input/intermediate/output tiles |
| `0x3000–0x37FF` | INT32 biases in accumulator units |
| `0x3800–0x38FF` | CFG: multiplier, shift, ReLU flag, reserved zero |
| `0x3900–0x3FFF` | Instructions; reset PC = `0x3900` |

Instruction layout: opcode `[127:124]`, src `[123:92]`, dest `[91:60]`, dims
`[59:32]`, flags `[31:0]`. Dims packs N in `[3:0]`, batch in `[7:4]`, K in
`[11:8]`. LOAD_OPERAND destination is `(slot << 2) | field`, with X=0, W=1,
B=2, CFG=3. Every load/store transfers exactly one 128-bit beat.

| Opcode | Instruction | Dispatch cycles, without back-pressure |
|---|---|---|
| `1` | STREAM_LOAD | 1 |
| `2` | EXEC_MAC | K |
| `3` | POST_PROC | 1 + POST_LATENCY |
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
consumes the completed accumulator tile; ReLU/bypass precedes INT64 multiply,
nearest-even shift and signed INT8 clamp. Inactive output bytes are zero.

This is a sequential instruction engine, not an overlapping issue pipeline.
Two operand slots do not automatically overlap loads with compute. Prefetch and
forwarding depend on the emitted program; obsolete `PREFETCH`/`FUSE` build flags
are rejected. For a 42-instruction 4->8->8->2 batch-4 program, the measured totals
are 266 cycles (POST_LATENCY=1) or 261 (0), with 32 compute clocks and 448 useful
MAC updates. Different instruction programs have different cycle counts.

Flow logs identify PC, opcode, k, active PEs, input, weight, product, prior sum,
and updated accumulator. Metrics include instructions, useful MACs, SRAM
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
connect its typed buses in `npu_top.c`, and register it in the block array.
All buses have one designated driver; consumers share that bus instance.

The callback registry, file loader, logs and host runner are simulation software.
They are not directly synthesizable HLS RTL; a hardware backend needs static
opcode decoding and explicit hardware interfaces with the same timing contract.

## Test

```sh
make -s test LOG=none
make -s test LOG=flow POST_LATENCY=0 SANITIZE=1

# Optional: compare with the existing compiler's emitted binaries and oracle.
make -s LOG=flow
python3 tests/check_model.py \
  --binary build/tb0-b1-post1-flow-san0/tiny_npu --compiler ../compiler
```

The built-in suite uses only Python's standard library. The optional compiler
comparison needs the compiler's Python dependencies and `LOG=flow` or `both`.
Tests cover all implemented opcodes, partial tiles, batches 1–4, retained sums,
forwarding, back-pressure, signed rounding, masked stores, malformed images,
overflow, and invalid instruction/state rejection. Expected fault tests print
error messages. On hosts that prohibit LeakSanitizer tracing, use
`ASAN_OPTIONS=detect_leaks=0`; address and undefined-behavior checks remain enabled.
