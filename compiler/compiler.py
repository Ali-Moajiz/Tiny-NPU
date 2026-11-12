"""Stages 3 and 5: map layers to hardware, then write the output binaries.

compile_model() emits typed instructions, not bytes. main.py passes these to
isa.encode_program(), then write_outputs() places them in the SRAM image.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from isa import (Instruction, Op, ENTRY_PC, SRAM_BYTES, REGIONS, ISA_VERSION,
                  dimensions, disassemble)


@dataclass
class Compilation:
    parameters: bytes
    instructions: list
    manifest: dict
    layers: list
    schedule: list

    def image(self, program):
        """Place the encoded program into its already-reserved SRAM region."""
        if len(program) != self.manifest['program_bytes']:
            raise ValueError("Encoded program length disagrees with reserved SRAM space")
        image = bytearray(self.parameters)
        image[ENTRY_PC:ENTRY_PC + len(program)] = program
        return bytes(image)


class Image:
    def __init__(self):
        self.data = bytearray(SRAM_BYTES)
        self.next = {name: start for name, (start, _) in REGIONS.items()}
        self.records = []

    def allocate(self, region, name, data):
        if not data or len(data) % 16:
            raise ValueError("Allocations must contain whole 128-bit beats")
        start = self.next[region]
        if start + len(data) > REGIONS[region][1]:
            raise ValueError(f"{region} capacity exceeded by {name} ({len(data)} bytes)")
        self.data[start:start + len(data)] = data
        self.next[region] += len(data)
        self.records.append(dict(name=name, region=region, address=start,
                                 address_hex=f"0x{start:05X}", size=len(data)))
        return start


def blocks(n):
    return (n + 3) // 4


def write_hex(path, data):
    # A line is bits [127:0]. Its rightmost byte is the lowest SRAM address.
    Path(path).write_text("".join(f"{int.from_bytes(data[i:i+16], 'little'):032X}\n"
                                 for i in range(0, len(data), 16)))


def compile_model(qlayers, batch=4, forward=True):
    """Shared backend for calibrated float layers and imported Q/DQ layers."""
    if batch not in range(1, 5):
        raise ValueError("Batch must be 1, 2, 3, or 4")
    if not qlayers:
        raise ValueError("Model must contain at least one quantized layer")
    image, instructions, schedule = Image(), [], []
    widths = [len(qlayers[0]['weights'][0])] + [len(q['weights']) for q in qlayers]
    buffers = [image.allocate("activations", f"tensor{i}", bytes(16 * blocks(w)))
               for i, w in enumerate(widths)]
    layouts = []
    for li, q in enumerate(qlayers):
        w, b = np.array(q["weights"], dtype=np.int8), np.array(q["bias"], dtype="<i4")
        n, k = w.shape
        wa, ba = [], []
        for nt in range(blocks(n)):
            row = []
            for kt in range(blocks(k)):
                tile = np.zeros((4, 4), dtype=np.int8)
                part = w[nt*4:nt*4+4, kt*4:kt*4+4]
                tile[:part.shape[0], :part.shape[1]] = part
                row.append(image.allocate("weights", f"L{li}.W[{nt},{kt}]", tile.tobytes()))
            wa.append(row)
            bt = np.zeros(4, dtype="<i4")
            part = b[nt*4:nt*4+4]
            bt[:len(part)] = part
            ba.append(image.allocate("bias", f"L{li}.B[{nt}]", bt.tobytes()))
        cfg = struct.pack("<IIII", q["multiplier"], q["shift"], int(q["activation"] == "relu"), 0)
        ca = image.allocate("config", f"L{li}.CFG", cfg)
        layouts.append(dict(weights=wa, bias=ba, config=ca))

    def emit(op, src=0, dest=0, dims=0, flags=0, note="", **debug):
        ins = Instruction(op, int(src), int(dest), int(dims), int(flags)).validate()
        instructions.append(ins)
        schedule.append(dict(pc=ENTRY_PC + 16*(len(instructions)-1),
                             instruction=ins.describe(), note=note, **debug))

    for kt in range(blocks(widths[0])):
        emit(Op.STREAM_LOAD, dest=buffers[0]+16*kt, dims=16, note="One padded input beat")
    useful_macs = 0
    forwarded = False
    for li, (q, layout) in enumerate(zip(qlayers, layouts)):
        n, k = len(q["weights"]), len(q["weights"][0])
        # Only a complete <=4-channel hidden tensor can be forwarded without
        # an assembler/gather buffer. Larger tensors use their SRAM tile layout.
        forward_next = forward and li + 1 < len(qlayers) and n <= 4
        for nt in range(blocks(n)):
            slot = (li + nt) & 1
            active_n = min(4, n-4*nt)
            emit(Op.LOAD_OPERAND, layout["bias"][nt], (slot << 2) | 2, 16, note="Bias in accumulator units")
            emit(Op.LOAD_OPERAND, layout["config"], (slot << 2) | 3, 16, note="Activation and scale configuration")
            for kt in range(blocks(k)):
                active_k = min(4, k-4*kt)
                emit(Op.LOAD_OPERAND, layout["weights"][nt][kt], (slot << 2) | 1, 16,
                     note=f"Layer {li}, neuron tile {nt}, reduction tile {kt}")
                # Forward lands in (li & 1); only nt=0 may reuse it. Later
                # neuron tiles reload from SRAM. Previous layer stores if needed.
                if not (forwarded and nt == 0 and kt == 0):
                    emit(Op.LOAD_OPERAND, buffers[li]+16*kt, slot << 2, 16, note="X[vector,k]")
                emit(Op.EXEC_MAC, dims=dimensions(active_n, batch, active_k),
                     flags=slot | (2 if kt else 0),
                     note="Retain accumulators" if kt else "Seed each vector accumulator with bias",
                     layer=li, neuron_offset=4*nt, k_offset=4*kt)
                useful_macs += active_n * batch * active_k
            # A hidden <=4 tensor may also be needed for extra next-layer neuron
            # tiles. Keep its SRAM copy in that case, in addition to forwarding.
            store = not forward_next or len(qlayers[li+1]["weights"]) > 4
            emit(Op.POST_PROC, dest=buffers[li+1]+16*nt if store else 0,
                 dims=dimensions(active_n, batch), flags=slot | (4 if store else 0),
                 note=f"Layer {li}: activation -> scale -> clamp", layer=li, neuron_offset=4*nt)
            if forward_next:
                emit(Op.FORWARD, dest=(li+1) & 1, dims=dimensions(active_n, batch),
                     note="Hidden result -> next layer slot.X")
        forwarded = forward_next
    for nt in range(blocks(widths[-1])):
        emit(Op.STREAM_STORE, src=buffers[-1]+16*nt, dims=16, note="One padded result beat")
    emit(Op.HALT, note="Finish after pending work and stream handshakes")
    # Reserve space now, so capacity errors happen before any files are written.
    program_bytes = 16 * len(instructions)
    image.allocate("program", "instructions", bytes(program_bytes))
    manifest = dict(isa=ISA_VERSION, entry_pc=ENTRY_PC, batch=batch, widths=widths,
                    input_scale=qlayers[0]["input_scale"], output_scale=qlayers[-1]["output_scale"],
                    buffers=buffers, layers=layouts, forwarding=forward,
                    instruction_count=len(instructions), program_bytes=program_bytes,
                    useful_macs_per_inference=useful_macs,
                    schedule_policy="conservative, dependency-ordered; no timing claim",
                    byte_order="little", beat_bytes=16, sram_bytes=SRAM_BYTES,
                    regions={k: {"start": v[0], "end_exclusive": v[1]} for k, v in REGIONS.items()},
                    allocations=image.records,
                    input_format="channel-tile-major beats, byte 4*v+k; inactive lanes zero",
                    output_format="channel-tile-major beats, byte 4*v+r; inactive lanes zero")
    return Compilation(bytes(image.data), instructions, manifest, qlayers, schedule)



# These files are optional, generated diagnostics; none drives hardware execution.
DEBUG_FILES = ("program.hex", "sram.hex", "program.asm", "manifest.json",
               "quantization.json", "schedule.json", "import_report.json", "memory_map.txt")


def write_outputs(compilation, program, output, report, source, debug=False):
    """Write three binaries, plus inspection/verification files only with --debug."""
    image = compilation.image(program)
    manifest = dict(compilation.manifest)
    manifest.update(quantization_mode=report['mode'],
                    source_input_dtype=report['source_input_dtype'],
                    source_output_dtype=report['source_output_dtype'],
                    calibration=report.get('calibration'), source=Path(source).name)
    manifest['sha256'] = {'program.bin': hashlib.sha256(program).hexdigest(),
                          'sram.bin': hashlib.sha256(image).hexdigest()}
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    for name, data in (("program.bin", program), ("params.bin", image[:ENTRY_PC]), ("sram.bin", image)):
        (target / name).write_bytes(data)
    # Never leave old diagnostic metadata describing a different binary.
    for name in (*DEBUG_FILES, "run.json"):
        (target / name).unlink(missing_ok=True)
    if debug:
        write_hex(target / "program.hex", program)
        write_hex(target / "sram.hex", image)
        (target / "program.asm").write_text(disassemble(program))
        for name, value in (("manifest.json", manifest), ("quantization.json", compilation.layers),
                            ("schedule.json", compilation.schedule), ("import_report.json", report)):
            (target / name).write_text(json.dumps(value, indent=2) + "\n")
        (target / "memory_map.txt").write_text("BYTE ADDR   SIZE  REGION       ALLOCATION\n" +
            "".join(f"0x{x['address']:05X}   {x['size']:6d}  {x['region']:<12} {x['name']}\n"
                    for x in manifest['allocations']))
    return manifest
