"""Stage 4: the NPU ISA. Opcode numbers, fields, validation and binary encoding.

Word: [127:124] opcode | [123:92] src | [91:60] dest |
      [59:32] dims | [31:0] flags. encode() writes 16 little-endian bytes.
Tile dims: [3:0] neurons, [7:4] vectors, [11:8] k; higher bits zero.
Read Op, Instruction.word(), then Instruction.encode() to understand encoding.
"""
from dataclasses import dataclass
from enum import IntEnum

ISA_VERSION = "tiny-npu-isa-v1"
ENTRY_PC = 0xF1000
SRAM_BYTES = 0x100000
REGIONS = {"weights": (0x00000, 0xC0000), "activations": (0xC0000, 0xE0000),
           "bias": (0xE0000, 0xF0000), "config": (0xF0000, 0xF1000),
           "program": (0xF1000, 0x100000)}
FIELDS = ("X", "W", "B", "CFG")


class Op(IntEnum):
    STREAM_LOAD = 1
    EXEC_MAC = 2
    POST_PROC = 3
    FORWARD = 4
    STREAM_STORE = 5
    LOOP_START = 6
    LOOP_END = 7
    LOAD_OPERAND = 8
    HALT = 15


def dimensions(neurons, vectors, k=0):
    return neurons | (vectors << 4) | (k << 8)


def unpack_dims(d):
    return d & 15, (d >> 4) & 15, (d >> 8) & 15


def address(addr, region):
    start, end = REGIONS[region]
    if addr % 16 or not start <= addr <= end - 16:
        raise ValueError(f"0x{addr:X}: expected aligned beat in {region} region")


@dataclass(frozen=True)
class Instruction:
    op: Op
    src: int = 0
    dest: int = 0
    dims: int = 0
    flags: int = 0

    def validate(self):
        Op(self.op)
        for val, bits in ((self.src, 32), (self.dest, 32), (self.dims, 28), (self.flags, 32)):
            if not isinstance(val, int) or not 0 <= val < (1 << bits):
                raise ValueError("Instruction field is outside its unsigned bit width")
        op = self.op
        n, v, k = unpack_dims(self.dims)
        if op in (Op.EXEC_MAC, Op.POST_PROC, Op.FORWARD):
            if self.dims >> 12 or not 1 <= n <= 4 or not 1 <= v <= 4:
                raise ValueError("Invalid tile dimensions")
            if (op == Op.EXEC_MAC and not 1 <= k <= 4) or (op != Op.EXEC_MAC and k):
                raise ValueError("Invalid k count")
        if op == Op.LOAD_OPERAND:
            if self.dest > 7 or self.flags or self.dims != 16:
                raise ValueError("LOAD_OPERAND requires slot/field DEST, dims=16, flags=0")
            address(self.src, ("activations", "weights", "bias", "config")[self.dest & 3])
        elif op == Op.STREAM_LOAD:
            if self.src or self.flags or self.dims != 16:
                raise ValueError("STREAM_LOAD requires FIFO_ID=0, one beat, flags=0")
            address(self.dest, "activations")
        elif op == Op.STREAM_STORE:
            if self.dest or self.flags or self.dims != 16:
                raise ValueError("STREAM_STORE requires FIFO_ID=0, one beat, flags=0")
            address(self.src, "activations")
        elif op == Op.EXEC_MAC:
            if self.src or self.dest or self.flags & ~3:
                raise ValueError("EXEC_MAC only accepts flags slot and accumulate")
        elif op == Op.POST_PROC:
            if self.src or self.flags & ~5:
                raise ValueError("POST_PROC only accepts flags slot and store")
            if self.flags & 4:
                address(self.dest, "activations")
            elif self.dest:
                raise ValueError("POST_PROC DEST must be zero without store")
        elif op == Op.FORWARD:
            if self.src or self.dest > 1 or self.flags:
                raise ValueError("FORWARD DEST selects slot 0 or 1")
        elif op == Op.LOOP_START:
            if not self.src or self.dims or self.flags:
                raise ValueError("LOOP_START needs positive count; dims/flags zero")
            address(self.dest, "program")
        elif op == Op.LOOP_END:
            if self.src or self.dims or self.flags:
                raise ValueError("LOOP_END src/dims/flags must be zero")
            address(self.dest, "program")
        elif op == Op.HALT and (self.src or self.dest or self.dims or self.flags):
            raise ValueError("HALT reserved fields must be zero")
        return self

    def word(self):
        self.validate()
        return (int(self.op) << 124) | (self.src << 92) | (self.dest << 60) | (self.dims << 32) | self.flags

    def encode(self):
        return self.word().to_bytes(16, "little")

    @classmethod
    def decode(cls, data):
        if len(data) != 16:
            raise ValueError("An instruction must contain exactly 16 bytes")
        word = int.from_bytes(data, "little")
        return cls(Op(word >> 124), (word >> 92) & 0xffffffff,
                   (word >> 60) & 0xffffffff, (word >> 32) & 0xfffffff,
                   word & 0xffffffff).validate()

    def describe(self):
        n, v, k = unpack_dims(self.dims)
        if self.op == Op.LOAD_OPERAND:
            detail = f"[0x{self.src:05X}] -> slot{self.dest >> 2}.{FIELDS[self.dest & 3]}"
        elif self.op == Op.EXEC_MAC:
            detail = f"slot={self.flags & 1} N={n} BATCH={v} K={k} accumulate={bool(self.flags & 2)}"
        elif self.op == Op.POST_PROC:
            detail = f"slot={self.flags & 1} N={n} BATCH={v} store={bool(self.flags & 4)} dest=0x{self.dest:05X}"
        elif self.op == Op.FORWARD:
            detail = f"result -> slot{self.dest}.X N={n} BATCH={v}"
        else:
            detail = f"src=0x{self.src:08X} dest=0x{self.dest:08X} dims={self.dims} flags=0x{self.flags:08X}"
        return f"OP_{self.op.name:<13} {detail}"


def disassemble(data, base=ENTRY_PC):
    if not data or len(data) % 16:
        raise ValueError("Program length must be a nonzero multiple of 16")
    return "\n".join(f"{base+i:05X}  {int.from_bytes(data[i:i+16], 'little'):032X}  "
                     + Instruction.decode(data[i:i+16]).describe()
                     for i in range(0, len(data), 16)) + "\n"


def encode_program(instructions):
    """Encode the scheduled instructions; no allocation or model parsing here."""
    if not instructions:
        raise ValueError("Program must contain instructions")
    return b"".join(instruction.encode() for instruction in instructions)
