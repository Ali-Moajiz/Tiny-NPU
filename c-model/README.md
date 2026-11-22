# Tiny NPU C-model

Run compiled Tiny-NPU SRAM images with cycle-level PE and bus logging.
Requires a C99 compiler, Make and Python 3 for tests.

## Run

From `Tiny-NPU/c-model`:

```sh
# 100 random batches, seed 42; match dimensions to the compiled model.
make -s run TB=1 BATCH=4 LOG=flow SRAM=../output/sram.hex \
  ARGS="--runs 100 --seed 42 --input-features 4 --output-features 2"

# Interactive input; q exits.
make -s run TB=1 BATCH=4 LOG=flow SRAM=../output/sram.hex \
  ARGS="--interactive --input-features 4 --output-features 2"

# Packed files, with the optional interactive/random TB excluded.
make -s run TB=0 BATCH=4 LOG=none SRAM=../output/sram.hex \
  ARGS="--input-bin input.bin --output-bin output.bin"
```

Inputs are **quantized INT8**, not floats. Apply the model's input scale before
supplying real-world values. `BATCH=1..4` must match the compiled instructions;
feature-count options control host packing and display, not network execution.
TB defaults are 100 batches, seed 42 and four input/output features. Each
inference resets the model and restores its preloaded SRAM image. `--input-bin`
runs once; `--output-bin` requires one inference. `--max-cycles` defaults to 100000.

A packed beat contains four channels for up to four vectors: byte
`4*vector + channel_in_tile`. Zero-pad unused vectors/channels. Larger tensors
use consecutive channel tiles. Outputs use the same layout.

Use `LOG=none|flow|ports|both`; append `2>&1 | tee npu.log` to save logs.
`SANITIZE=1` enables ASan/UBSan. `POST_LATENCY=1` is a deprecated compatibility
argument accepted by existing scripts; it does not change timing.
`POST_LATENCY=0`, `PREFETCH` and `FUSE` build switches are unsupported. The
instruction stream determines operand loads and forwarding.

## File layout

**24 maintained files: 18 implementation files, 4 tests, Makefile and README.**
Headers are in `include/`, implementations in `src/`, and tests in `tests/`.
Build products live in `build/` and are not source files. Run Make from the
`c-model/` directory; its include search path is configured automatically.

| Files | Responsibility |
|---|---|
| `src/npu.c`, `include/npu.h` | Top-level instances, wiring, initialization and cycle entry point |
| `src/controller.c`, `include/controller.h` | ISA, instruction handlers and fetch/decode/dispatch/update FSM |
| `src/sram.c`, `include/sram.h` | Single-port SRAM and registered read response |
| `src/mac.c`, `include/mac.h` | MAC PE datatype, multiplier/adder/accumulator, operand slots and 4x4 array |
| `src/activation.c`, `include/activation.h` | Activation functions, registered PE and four lane instances |
| `src/scale_clamp.c`, `include/scale_clamp.h` | Registered scaling PE, four lanes and result packer |
| `include/types.h` | Shared widths, memory map, bus payloads and integer rounding |
| `src/bus.c`, `include/bus.h` | Typed bus API, ownership checks, block registration and simulation clock phases |
| `src/sim.c`, `include/sim.h` | Host image/file I/O, stream endpoint, logging, metrics and optional TB helpers |
| `src/main.c` | Host command-line runner |
| `tests/check_units.c` | Bus, controller, memory, arithmetic and PE timing/fault tests |
| `tests/check_image.c` | Image-loader regression helper |
| `tests/check_model.py` | Independent numerical, cycle and compiler-integration checks |
| `tests/check_tb.py` | Random/interactive TB regression checks |

Start with `include/npu.h` and `src/npu.c` to see the complete wiring. Then read each block's
header for its registers/ports and its `.c` file for behavior. PE definitions
and their parent arrays stay together; there are no separate PE wrapper files.

## Registers, buses and clocks

Each PE owns `regs` and `ports`. Combinational scratch is named `comb`; the MAC
also exposes its combinational `enabled` signal. There are no q/d register copies.
Only initialization connects ports. Consumers share the producer's bus object.
These buses model wires, not AXI transactions.

Every simulated cycle runs **publish → evaluate → check → clock**. All old
registered outputs are published before evaluation. Combinational dependencies
must be evaluated in order; registered-stage order does not affect their data.
Every check completes before any register changes. Invalid transfers and INT32
overflow reject the whole edge, preventing partial updates.

The 16 cells are `mac.cell[vector][neuron]`. A row shares X; a column shares W
and bias. Each cell owns its accumulator and result bus. At each k-step:

`acc = (bias if k=0 and !accumulate else acc) + X[vector][k] * W[neuron][k]`

The signed product is 16 bits; accumulation is INT32 with an overflow check.
Disabled cells hold state. Two operand slots inside the MAC hold X/W/B/CFG.
They do not automatically overlap loads with compute: instructions issue
sequentially. Retained sums survive operand loads and further MAC instructions.

POST reads one neuron column per clock into four activation lanes (one lane
per vector). Activation and scaling each have one output register. A packer
collects INT8 columns, zero-pads inactive bytes and holds the completed 128-bit
result for SRAM or forwarding. Forwarding copies the packed tile without a
transpose; there is no result FIFO.

| POST clock, N=4 | Activation capture | Scaling capture | Packer / store |
|---|---|---|---|
| T | Column 0 | — | Clear tile |
| T+1 | Column 1 | Column 0 | — |
| T+2 | Column 2 | Column 1 | Column 0 |
| T+3 | Column 3 | Column 2 | Column 1 |
| T+4 | — | Column 3 | Column 2 |
| T+5 | — | — | Column 3; result complete |
| T+6 | — | — | Optional SRAM store |

POST dispatch takes `N+2+store` clocks, where store is 0 or 1. A value captured
at clock C is available in clock C+1. Flow logs label PC, k, PE coordinates,
products, partial sums and ACT/SCALE/PACK events. Metrics count useful MACs,
compute clocks, SRAM accesses and utilization. Bus stalls count additional
waits for unavailable input/output/response, not normal fetch latency.

The golden batch-4 test computes at clocks 33–36, reads columns at 41–44,
packs the last column at 46, stores at 47 and halts at 59. The compiler's
42-instruction 4→8→8→2 regression takes 289 clocks, 32 compute clocks and 448
useful MAC updates. This file consolidation preserves those timings.

## Memory and ISA

SRAM is one 1 MiB array, 65,536 × 128 bits. Byte addresses are 16-byte aligned.
`sram.hex` must contain exactly 65,536 words of 32 hex digits: bits 127..0,
lowest-address byte at the right. Blank lines and `#`/`//` comments are accepted.
The host loads and validates the complete image once; simulated reads access
only the SRAM array. `program.hex` alone is not a complete image.

| Byte range (inclusive) | Contents |
|---|---|
| `0x00000–0xBFFFF` | INT8 weights |
| `0xC0000–0xDFFFF` | Input/intermediate/output tiles |
| `0xE0000–0xEFFFF` | INT32 biases in accumulator units |
| `0xF0000–0xF0FFF` | Quantization/activation configuration |
| `0xF1000–0xFFFFF` | Instructions; reset PC `0xF1000` |

Instruction fields are opcode `[127:124]`, source `[123:92]`, destination
`[91:60]`, dimensions `[59:32]`, flags `[31:0]`. Dimensions pack N in `[3:0]`,
batch in `[7:4]`, K in `[11:8]`. LOAD_OPERAND destination is `(slot<<2)|field`,
where X=0, W=1, B=2, CFG=3. Each transfer moves one 128-bit beat.

| Opcode | Instruction | Dispatch clocks, excluding back-pressure |
|---|---|---|
| 1 | STREAM_LOAD | 1 |
| 2 | EXEC_MAC | K |
| 3 | POST_PROC | N+2+store |
| 4 | FORWARD | 1 |
| 5 | STREAM_STORE | 2 |
| 6, 7 | LOOP_START, LOOP_END | 1 each |
| 8 | LOAD_OPERAND | 2 |
| F | HALT | 1 |

Each instruction also costs four lifecycle clocks: fetch request, response,
decode and update. EXEC_MAC flag bit 0 selects the slot; bit 1 retains sums.
POST flag bit 2 enables store. One non-nested hardware loop is supported.
The instruction region holds 3840 words. Host stream capacity is 256 beats
(up to 1024 features at batch 4). Array size and compiler packing remain 4x4.

CFG is four little-endian 32-bit words: positive INT32 multiplier M, shift
S=0..62, activation mode and parameter. Activation precedes multiplication by
M, nearest-even rounded shift S and saturation to [-128,127].

| Mode | Operation | Parameter |
|---|---|---|
| 0 | Bypass | 0 |
| 1 | ReLU | 0 |
| 2 | LeakyReLU: negative branch round-even(x / 2^p) | p=0..31 |
| 3 | Clipped ReLU: min(C,max(0,x)) | positive INT32 C |

Parameters use accumulator units. C=6 is not automatically floating-point
ReLU6. Existing compiler binaries use modes 0/1; modes 2/3 await compiler
support. Tanh/sigmoid require a specified numeric format and approximation
budget and are not silently implemented using host floating-point functions.

## HLS boundary

This is a **cycle-accurate C reference model, not a synthesis-qualified full
NPU top**. File consolidation alone cannot make the simulation framework RTL.

The hardware arithmetic uses fixed-width integers, static arrays, bounded
loops and explicit register boundaries. `mac_pe_datapath()`,
`activation_value()` and `scale_clamp_value()` isolate scalar arithmetic from
bus traversal and host I/O. Rounded division by a power of two uses unsigned
shifts and masks. No floating-point library is used in the PEs.

For full-core HLS, the backend must replace simulation callback dispatch and
checked pointer buses with static hardware calls/interfaces, replace the host
stream endpoint, and select SRAM/interface/pipeline directives for the target
tool. Diagnostic calls and `src/main.c`/`src/sim.c` stay outside the synthesis graph.
HLS scheduling may add stages: verify generated RTL latency and resource
sharing against this model. No HLS synthesis or RTL equivalence run is claimed.

## Extend and test

Add an opcode's validator/execution callback and descriptor in `src/controller.c`;
`controller_register()` still supports extensions without changing the FSM.
Callbacks emit control signals; they do not directly modify datapath registers.
For a new block, define its registers/ports and `BlockOps`, connect its buses
in `src/npu.c`, increase registry capacity if needed and register the instance once.

```sh
make -s test LOG=none
make -s test LOG=both SANITIZE=1

# Compiler dependencies are needed for these optional comparisons.
make -s LOG=flow
python3 tests/check_model.py \
  --binary build/tb0-b1-simple-flow-san0/tiny_npu --compiler ../compiler
make -s TB=1 BATCH=4 LOG=flow
python3 tests/check_tb.py \
  --binary build/tb1-b4-simple-flow-san0/tiny_npu --compiler ../compiler
```

Tests include masked writes, malformed images, bus ownership, atomic faults,
PE timing, all activation modes, partial tiles, batches 1–4, retained sums,
forwarding, back-pressure and independent per-k/output oracles. To compare a
pre-consolidation PE executable, pass `--baseline-binary /path/to/tiny_npu` to
`check_model.py` together with `--compiler`; this also compares full flow logs
and cycle counts. Expected-fault tests print errors. On hosts that cannot run
LeakSanitizer under tracing, use `ASAN_OPTIONS=detect_leaks=0`; ASan/UBSan remain
active. The compiler and architecture diagram are separate project directories.
