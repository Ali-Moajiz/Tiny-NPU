"""Optional binary verification, not part of model compilation.

1. execute() runs the actual SRAM/instruction bytes, with per-k traces.
2. oracle() independently evaluates integer tensors with exact Fraction rounding.
3. run_directory() compares both, including every accumulator and final output.
This is functional verification, NOT a cycle-accurate hardware model.
"""
from fractions import Fraction
import json
from pathlib import Path
import struct
import numpy as np
from isa import Instruction, Op, ENTRY_PC, SRAM_BYTES, ISA_VERSION, address, unpack_dims
from quantization import int8, requantize


def pack_tensor(x):
    x = np.asarray(x)
    if x.ndim != 2 or not 1 <= x.shape[0] <= 4 or x.shape[1] < 1:
        raise ValueError("Tensor must be [batch 1..4, channels >=1]")
    if not np.isfinite(x).all() or np.any(x != np.rint(x)) or np.any(x < -128) or np.any(x > 127):
        raise ValueError("Expected integer INT8 tensor")
    beats = []
    for k in range(0, x.shape[1], 4):
        tile = np.zeros((4, 4), dtype=np.int8)
        part = x[:, k:k+4]
        tile[:len(x), :part.shape[1]] = part
        beats.append(tile.tobytes())
    return b"".join(beats)

def unpack_tensor(data, batch, channels):
    if len(data) != 16*((channels+3)//4):
        raise ValueError("Unexpected output FIFO byte count")
    tiles = [np.frombuffer(data[i:i+16], dtype=np.int8).reshape(4, 4)[:batch]
             for i in range(0, len(data), 16)]
    return np.concatenate(tiles, axis=1)[:, :channels].copy()

def execute(image, input_bytes, entry_pc=ENTRY_PC, max_instructions=100000):
    if len(image) != SRAM_BYTES or len(input_bytes) % 16:
        raise ValueError(f"Expected {SRAM_BYTES} SRAM bytes and whole input beats")
    mem = bytearray(image)
    slots = [[None] * 4 for _ in range(2)]
    acc = np.zeros((4, 4), dtype=np.int64)  # [neuron, vector]
    acc_dims, result, result_dims, loop = None, None, None, None
    fifo_pos, pc, count, useful = 0, entry_pc, 0, 0
    output, trace = bytearray(), []
    while count < max_instructions:
        address(pc, "program")
        ins = Instruction.decode(mem[pc:pc+16])
        count += 1
        next_pc = pc + 16
        op, slot = ins.op, ins.flags & 1
        n, v, k_count = unpack_dims(ins.dims)
        event = dict(instruction=count, pc=pc, operation=ins.describe())
        if op == Op.STREAM_LOAD:
            if fifo_pos + 16 > len(input_bytes):
                raise ValueError("Input FIFO underflow")
            mem[ins.dest:ins.dest+16] = input_bytes[fifo_pos:fifo_pos+16]
            fifo_pos += 16
        elif op == Op.LOAD_OPERAND:
            slots[ins.dest >> 2][ins.dest & 3] = bytes(mem[ins.src:ins.src+16])
        elif op == Op.EXEC_MAC:
            values = slots[slot]
            if values[0] is None or values[1] is None or (not ins.flags & 2 and values[2] is None):
                raise ValueError("MAC issued with invalid operands")
            x = np.frombuffer(values[0], dtype=np.int8).reshape(4, 4)
            w = np.frombuffer(values[1], dtype=np.int8).reshape(4, 4)
            if ins.flags & 2:
                if acc_dims != (n, v):
                    raise ValueError("Accumulate requires a live tile of the same dimensions")
            else:
                b = np.frombuffer(values[2], dtype="<i4")
                acc.fill(0)
                acc[:n, :v] = b[:n, None]
                acc_dims = (n, v)
            steps = []
            for k in range(k_count):
                # Each PE owns its accumulator, no horizontal reduction chain.
                for r in range(n):
                    for c in range(v):
                        value = int(acc[r, c]) + int(w[r, k]) * int(x[c, k])
                        if not -2**31 <= value < 2**31:
                            raise ValueError("INT32 accumulator overflow")
                        acc[r, c] = value
                useful += n*v
                steps.append(dict(local_k=k, active_pes=n*v, accumulators=acc[:n, :v].tolist()))
            event["steps"] = steps
        elif op == Op.POST_PROC:
            if acc_dims != (n, v) or slots[slot][3] is None:
                raise ValueError("POST needs valid accumulators and config")
            m, s, act, reserved = struct.unpack("<IIII", slots[slot][3])
            if not 1 <= m <= 0x7fffffff or s > 62 or act > 1 or reserved:
                raise ValueError("Invalid quantization configuration")
            y = np.zeros((4, 4), dtype=np.int8)  # [vector, neuron]
            for r in range(n):
                for c in range(v):
                    y[c, r] = requantize(int(acc[r, c]), m, s, act)
            result, result_dims = y.tobytes(), (n, v)
            event["output"] = y[:v, :n].tolist()
            acc_dims = None
            if ins.flags & 4:
                mem[ins.dest:ins.dest+16] = result
        elif op == Op.FORWARD:
            if result is None or result_dims != (n, v):
                raise ValueError("FORWARD needs a matching completed result")
            slots[ins.dest][0] = result
        elif op == Op.STREAM_STORE:
            output.extend(mem[ins.src:ins.src+16])
        elif op == Op.LOOP_START:
            if loop is not None or ins.dest != pc+16:
                raise ValueError("Only a single non-nested loop with body at PC+16 is supported")
            loop = [ins.src, ins.dest]
        elif op == Op.LOOP_END:
            if loop is None or ins.dest != loop[1]:
                raise ValueError("Unmatched LOOP_END")
            loop[0] -= 1
            if loop[0]:
                next_pc = loop[1]
            else:
                loop = None
        elif op == Op.HALT:
            if loop is not None:
                raise ValueError("HALT inside active loop")
            if fifo_pos != len(input_bytes):
                raise ValueError("Input FIFO has unused bytes")
            trace.append(event)
            return dict(output=bytes(output), sram=bytes(mem), trace=trace,
                        instructions=count, useful_macs=useful)
        trace.append(event)
        pc = next_pc
    raise ValueError("Instruction limit reached (missing HALT or runaway loop)")

def oracle(qlayers, inputs):
    x = np.asarray(inputs, dtype=np.int64)
    layer_outputs, partials = [], []
    for li, q in enumerate(qlayers):
        w = np.array(q["weights"], dtype=np.int64)
        b = np.array(q["bias"], dtype=np.int64)
        acc = np.tile(b, (len(x), 1))
        for k in range(w.shape[1]):
            acc = acc + x[:, k:k+1] * w[:, k]
            if (acc < -2**31).any() or (acc > 2**31-1).any():
                raise ValueError("Reference INT32 overflow")
            partials.append(dict(layer=li, k=k, accumulators=acc.T.tolist()))
        y = np.empty(acc.shape, dtype=np.int8)
        for v in range(len(x)):
            for n in range(len(b)):
                a = int(acc[v, n])
                if q["activation"] == "relu":
                    a = max(a, 0)
                # Fraction.__round__ supplies independent exact ties-to-even.
                scaled = round(Fraction(a * q["multiplier"], 1 << q["shift"]))
                y[v, n] = max(-128, min(127, scaled))
        x = y.astype(np.int64)
        layer_outputs.append(y.tolist())
    return layer_outputs, partials

def run_directory(directory, inputs, quantized=False):
    root = Path(directory)
    if not (root / "manifest.json").is_file():
        raise ValueError("Verification needs metadata: recompile the model with --debug, then run again")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["isa"] != ISA_VERSION:
        raise ValueError("Unsupported ISA version")
    qlayers = json.loads((root / "quantization.json").read_text())
    schedule = {x["pc"]: x for x in json.loads((root / "schedule.json").read_text())}
    image = (root / "sram.bin").read_bytes()
    x = np.asarray(inputs)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] != manifest["widths"][0] or not np.isfinite(x).all():
        raise ValueError("Inputs must be finite [samples, input_features]")
    if quantized:
        if np.any(x != np.rint(x)) or np.any(x < -128) or np.any(x > 127):
            raise ValueError("--quantized requires integer values in [-128,127]")
        xq = x.astype(np.int8)
    else:
        xq = int8(x, manifest["input_scale"])
    outputs, runs = [], []
    batch = manifest["batch"]
    for start in range(0, len(xq), batch):
        real = min(batch, len(xq)-start)
        tile = np.zeros((batch, xq.shape[1]), dtype=np.int8)
        tile[:real] = xq[start:start+real]
        result = execute(image, pack_tensor(tile), manifest["entry_pc"])
        expected_layers, partials = oracle(qlayers, tile)
        expected_by_step = {(p["layer"], p["k"]): p["accumulators"] for p in partials}
        for event in result["trace"]:
            meta = schedule[event["pc"]]
            if "steps" in event:
                li, offset = meta["layer"], meta["neuron_offset"]
                for step in event["steps"]:
                    k = meta["k_offset"] + step["local_k"]
                    actual = step["accumulators"]
                    expected = expected_by_step[li, k][offset:offset+len(actual)]
                    if actual != expected:
                        raise ValueError(f"Per-k mismatch: L{li}, neuron={offset}, k={k}")
                event.update(layer=li, neuron_offset=offset, k_offset=meta["k_offset"])
            if "output" in event:
                li, offset = meta["layer"], meta["neuron_offset"]
                expected = np.array(expected_layers[li])[:, offset:offset+len(event["output"][0])].tolist()
                if event["output"] != expected:
                    raise ValueError(f"Post-process mismatch at L{li}, neuron={offset}")
        actual = unpack_tensor(result["output"], batch, manifest["widths"][-1])
        if actual.tolist() != expected_layers[-1]:
            raise ValueError("Binary output differs from integer oracle")
        outputs.extend(actual[:real].tolist())
        runs.append(dict(first_sample=start, actual_samples=real, padded_samples=batch-real,
                         instructions=result["instructions"], useful_macs=result["useful_macs"],
                         trace=result["trace"]))
    return dict(isa=ISA_VERSION, checked="per-k accumulators, every post tile, final outputs",
                input_int8=xq.tolist(), output_int8=outputs,
                output_scale=manifest["output_scale"],
                output_float=(np.array(outputs)*manifest["output_scale"]).tolist(),
                runs=runs, timing="Functional instruction trace; no clock-cycle estimates")
