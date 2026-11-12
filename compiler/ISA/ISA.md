# Tiny NPU ISA v1 — compiler / decoder contract

This document fixes the previously unspecified encodings for the current
single-SRAM architecture. Version identifier `tiny-npu-isa-v1` lives in
`manifest.json`. The raw instruction stream has no header and no embedded
version. A loader must verify the manifest version before starting the decoder.

## Instruction layout

| Bits | Width | Field |
|---|---:|---|
| 127:124 | 4 | Opcode |
| 123:92 | 32 | SRC / immediate / loop count |
| 91:60 | 32 | DEST / target / operand latch selector |
| 59:32 | 28 | DIMS / transfer byte count |
| 31:0 | 32 | FLAGS |

All unused bits/fields must be zero. The ISA is unsigned at the instruction
field level. Signed values are in parameter SRAM. `program.bin` stores each word
little-endian. Each `.hex` line prints `[127:0]`, most significant digit first.
The rightmost byte on a hex line is the byte at the lowest memory address.

For compute/post/forward DIMS is:

| DIMS-relative bits | Instruction bits | Meaning |
|---|---|---|
| 3:0 | 35:32 | N, active neurons, 1..4 |
| 7:4 | 39:36 | V, active independent vectors, 1..4 |
| 11:8 | 43:40 | K, reduction steps, 1..4 for EXEC; zero for POST/FORWARD |
| 27:12 | 59:44 | Reserved, zero |

For transfer instructions DIMS is exactly 16, meaning **one 128-bit beat**.
There are no bursts, automatic strides or implicit address increments in v1.
The compiler emits each address explicitly. PC advances by 16 bytes unless a
loop branches. All PCs and SRAM transfer addresses must be 16-byte aligned.

## Opcode map

| Hex | Name | SRC | DEST | DIMS | FLAGS |
|---|---|---|---|---|---|
| 1 | OP_STREAM_LOAD | Input FIFO ID, 0 | Activation SRAM byte address | 16 | 0 |
| 2 | OP_EXEC_MAC | 0 | 0 | N,V,K | bit0 slot; bit1 accumulate |
| 3 | OP_POST_PROC | 0 | Store byte address, or 0 | N,V,0 | bit0 CFG slot; bit2 store |
| 4 | OP_FORWARD | 0 | Destination slot 0/1 | N,V,0 | 0 |
| 5 | OP_STREAM_STORE | Activation SRAM byte address | Output FIFO ID, 0 | 16 | 0 |
| 6 | OP_LOOP_START | Positive iteration count | Body PC, must equal current PC+16 | 0 | 0 |
| 7 | OP_LOOP_END | 0 | Matching body PC | 0 | 0 |
| 8 | OP_LOAD_OPERAND | SRAM byte address | bit2 slot; bits1:0 field | 16 | 0 |
| F | OP_HALT | 0 | 0 | 0 | 0 |

Operand latch field codes: X=0, W=1, B=2, CFG=3. There are two slots, each with
one 128-bit field of each kind. `DEST=(slot<<2)|field`. OP_LOAD_OPERAND copies
one SRAM beat directly into that field, independent of the CPU data path.
The opcode is 0x8 as proposed in the diagram. No existing opcode is renumbered.
Opcodes 0,9,A,B,C,D,E are reserved and rejected.

## Instruction behavior

- **STREAM_LOAD:** consume exactly one FIFO beat into SRAM. Stream payloads
  already contain quantized INT8 input values in the packed layout below.
- **EXEC_MAC:** read selected X and W. If accumulate=0, initialize every active
  PE accumulator to its neuron bias from B, then execute K outer-product steps.
  If accumulate=1, retain the existing N×V accumulator tile and add K steps.
  X/W/B remain in registers after use. Accumulator storage is shared by both
  operand slots, so two MAC tiles cannot execute concurrently.
- **POST_PROC:** all active accumulator lanes pass through the **activation
  block**, then the separate **scale/clamp block**. The result register gets
  one 128-bit padded tile. An optional store copies it to SRAM. The accumulator
  tile is consumed and cannot be accumulated again without a new EXEC start.
- **FORWARD:** copy the last matching post result into destination slot.X.
  Result byte layout already matches X. The result can be both stored and
  forwarded. There is no forwarding of wide unquantized accumulators.
- **STREAM_STORE:** read one padded result beat from SRAM and push it to the
  output FIFO. The receiver uses manifest dimensions to discard padding.
- **LOOP_START/END:** one non-nested loop context. START sets remaining count
  and body PC. END decrements and branches while nonzero. No pointer register
  arithmetic or strides are implied. The compiler currently unrolls tiles.
- **HALT:** real hardware waits for all pending operations/stores and output
  handshakes, then signals completion. The functional interpreter has no
  outstanding operations by construction.

## Hardware dependency contract

The controller follows Fetch → Decode → Dispatch/wait → Update. Future
hardware may update PC on command acceptance. It must stall dispatch when a
source is not valid, a target slot field is still being read, the accumulator
or result register is occupied, a stream handshake cannot complete, or the
single SRAM port is busy. The current interpreter completes instructions in
order, which supplies the architectural reference behavior.

Do not overwrite a slot's CFG until its post operation has consumed it. Do not
overwrite X/W while its MAC operation is active. A new MAC operation must wait
until prior post-processing has released shared accumulators. A new post result
must wait until prior forwarding/store has consumed the previous result.

No explicit BUS_WAIT is introduced in this v1: the latest diagram's
dependency-based Dispatch/wait contract supplies the wait. The older external
bus variant is not the target here. Scheduling/compiler instruction counts
are not clock counts. Latency-aware prefetch and hazard simulation are future
work; no bandwidth or cycle improvement is claimed by this initial compiler.

## SRAM and packing

| Byte address range | Use | Capacity |
|---|---|---:|
| 00000–BFFFF | Weight tiles | 768 KiB |
| C0000–DFFFF | Input, hidden, final tensor tiles | 128 KiB |
| E0000–EFFFF | Bias tiles | 64 KiB |
| F0000–F0FFF | Per-layer config beats | 4 KiB |
| F1000–FFFFF | Instruction program, entry PC=F1000 | 60 KiB / 3840 words |

This is one physical 65,536×128 SRAM (1 MiB). Byte-address bits [19:4] select
its line. ISA address fields remain 32 bits. Regions have exclusive ends
C0000, E0000, F0000, F1000, and 100000 respectively.
External CSR/SoC interfaces are outside this package and are not emulated.

- **Weight tile:** byte `4*r+k` = signed INT8 `W[neuron_offset+r,k_offset+k]`.
- **X tile:** byte `4*v+k` = signed INT8 `X[vector=v,k_offset+k]`.
- **Bias tile:** signed little-endian INT32 at bytes `4*r..4*r+3` = B[neuron+r].
- **Post/result tile:** byte `4*v+r` = signed INT8 Y[vector=v,neuron_offset+r].
- Tensor tiles in SRAM/FIFOs are channel-tile-major. Within one tile, each
  vector has four consecutive bytes. Inactive rows/columns are zero. Every
  transfer is a complete beat, so no byte-enable subfield is needed by this
  compiler v1. A future masked-transfer extension would need a versioned spec.

CFG is four little-endian uint32 words:

| Bytes | Bits | Meaning |
|---|---|---|
| 0..3 | 31:0 | M, range 1..2^31−1 |
| 4..7 | 63:32 | S, range 0..62 |
| 8..11 | 95:64 | Activation, 0=bypass, 1=ReLU |
| 12..15 | 127:96 | Reserved=0 |

Scales for input conversion/output interpretation are host metadata in
`manifest.json`. Hardware post-processing needs only the packed integer M/S
configuration. Bias is already quantized to accumulator units. Dequantization
uses the fixed output scale. Hardware needs no floating-point computation.

## Implementation choices versus the earlier diagram

The diagram fixed outer fields, opcodes, blocks and memory regions. This version
defines DIMS nibbles, flag masks, config word layout, byte ordering, full-beat
padding, positive M with S≤62, the loop contract, and dependency validity rules.
It conservatively schedules instructions and rejects unsupported operators or
capacity overflow. These details must be implemented in a future C decoder;
binary compatibility with the previous static-schedule model is not claimed.