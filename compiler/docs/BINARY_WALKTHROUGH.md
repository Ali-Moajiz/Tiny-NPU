# Inspect the included 4→4→4 binary

Open `compiled_example/program.asm`. It decodes the actual `program.bin` bytes.
Addresses shown at the left are **byte PCs**, not cycle numbers. One instruction
can require many hardware clocks.

## Read the program in execution order

| PC | Instruction | Effect |
|---|---|---|
| 3900 | STREAM_LOAD | Runtime input beat → SRAM[2000] |
| 3910 | LOAD_OPERAND | SRAM[3000] → slot0.B |
| 3920 | LOAD_OPERAND | SRAM[3800] → slot0.CFG |
| 3930 | LOAD_OPERAND | SRAM[0000] → slot0.W |
| 3940 | LOAD_OPERAND | SRAM[2000] → slot0.X |
| 3950 | EXEC_MAC | Layer 1, 4 neurons × 4 vectors × 4 k-steps |
| 3960 | POST_PROC | ReLU then scale/clamp, result register only |
| 3970 | FORWARD | Quantized hidden tile → slot1.X |
| 3980 | LOAD_OPERAND | SRAM[3010] → slot1.B |
| 3990 | LOAD_OPERAND | SRAM[3810] → slot1.CFG |
| 39A0 | LOAD_OPERAND | SRAM[0010] → slot1.W |
| 39B0 | EXEC_MAC | Layer 2, 4 neurons × 4 vectors × 4 k-steps |
| 39C0 | POST_PROC | Activation bypass then scale/clamp → SRAM[2020] |
| 39D0 | STREAM_STORE | SRAM[2020] → output FIFO |
| 39E0 | HALT | Complete execution |

All addresses in this table are hexadecimal. The hidden region at 2010 is
reserved but unused in this forwarded example. It receives no hidden-result
write or subsequent input read. `--no-forward` uses that SRAM region instead.

This program performs 128 useful MAC updates, and contains 15 instructions.
It does **not** take 15 clocks: instruction fetching shares the SRAM port,
the controller has decode/dispatch overhead, and each EXEC has four k-steps.
The functional runner makes no total-cycle prediction.

## Decode one MAC instruction by hand

At PC 3950 the hex word is:

```text
20000000000000000000044400000000
```

| Field | Value | Interpretation |
|---|---|---|
| opcode | 2 | OP_EXEC_MAC |
| src | 00000000 | Reserved zero |
| dest | 00000000 | Reserved zero |
| dims | 0000444 | K=4, V=4, N=4 |
| flags | 00000000 | Slot0, start from bias |

To retrieve a field from the unsigned 128-bit integer `word`:

```python
opcode = (word >> 124) & 0xF
src    = (word >> 92)  & 0xFFFFFFFF
dest   = (word >> 60)  & 0xFFFFFFFF
dims   = (word >> 32)  & 0x0FFFFFFF
flags  = word         & 0xFFFFFFFF
```

The high hex digits appear last in the little-endian `.bin` byte stream.
Use `int.from_bytes(instruction_bytes, "little")` before applying the masks.

## Understand one PE

PE[r,v] computes one neuron's output for one independent vector. For EXEC at
3950, all 16 PEs operate on each k-step:

| Step | PE[2,1] update |
|---|---|
| Initialization | acc[2,1] = bias[2] |
| k=0 | acc[2,1] += W[2,0] × X[1,0] |
| k=1 | acc[2,1] += W[2,1] × X[1,1] |
| k=2 | acc[2,1] += W[2,2] × X[1,2] |
| k=3 | acc[2,1] += W[2,3] × X[1,3] |

In `compiled_example/run.json`, find `pc: 14672` (0x3950). Its `steps` array
contains all four per-k accumulator snapshots, indexed `[neuron][vector]`.
Activation happens in the next POST instruction, after accumulation is complete.
The trace's `instruction` field counts dynamic instructions, not clock cycles.

## Inspect the weights and SRAM image

1. Use `memory_map.txt` to find a parameter's byte address.
2. Divide that address by 16 to obtain a zero-based `sram.hex` line index.
3. Read each line as a 128-bit word. Byte lane 0 is the rightmost byte.
4. For W, bytes `4*r+k` hold INT8 W[r,k]. Interpret 80..FF as negative two's
   complement numbers. For bias, group every four bytes as little-endian INT32.
5. `quantization.json` provides the same integers in a readable list, plus
   their input/weight/output scales.

At compile time the input/result regions in `sram.bin` are zero. The `run`
command supplies runtime input separately through the simulated input FIFO.

## Expected example output

Inputs are floating-point features; they are first quantized with the compiled
input scale. The example outputs are:

```text
[1, 2, 3, 4]    -> [27, 21, 2, 57]
[0, 1, -4, -1]  -> [0, -2, -1, 0]
[0, 0, 0, 0]    -> [1, -1, -1, 3]
[-1, 2, -3, 4]  -> [0, -2, -2, 1]
```

These values belong to the included deterministic toy weights. They are not
predictions for your trained classifier. Compile your exported model and real
calibration data to generate its own parameters and binary.
