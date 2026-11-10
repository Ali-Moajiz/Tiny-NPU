import json
from pathlib import Path
import struct
import tempfile
import unittest
from fractions import Fraction
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator
from tiny_npu.compiler import build, compile_model
from tiny_npu.cli import run_directory
from tiny_npu.frontend import load_model
from tiny_npu.ir import Layer, run_float
from tiny_npu.isa import Instruction, Op, ENTRY_PC, SRAM_BYTES, dimensions, disassemble
from tiny_npu.interpreter import execute, pack_tensor, unpack_tensor
from tiny_npu.quantize import round_shift, quantize, requantize
from tiny_npu.reference import oracle


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def save_json(path, layers):
    path.write_text(json.dumps(dict(format="tiny-npu-float-v1", layers=[
        dict(name=l.name, weights=l.weights.tolist(), bias=l.bias.tolist(), activation=l.activation)
        for l in layers])))


def program_image(instructions):
    image = bytearray(SRAM_BYTES)
    data = b"".join(x.encode() for x in instructions)
    image[ENTRY_PC:ENTRY_PC+len(data)] = data
    return image


class EncodingTests(unittest.TestCase):
    def test_golden_encoding_and_endianness(self):
        i = Instruction(Op.LOAD_OPERAND, src=0x2010, dest=4, dims=16)
        # Construct expected bit-string independently, field by field.
        bits = "1000" + f"{0x2010:032b}" + f"{4:032b}" + f"{16:028b}" + "0"*32
        golden = int(bits, 2)
        self.assertEqual(i.word(), golden)
        self.assertEqual(i.encode(), golden.to_bytes(16, "little"))
        self.assertEqual(Instruction.decode(i.encode()), i)
        self.assertEqual(Instruction(Op.HALT).encode(), b"\0"*15+b"\xf0")

    def test_all_opcodes_roundtrip(self):
        examples = [Instruction(Op.STREAM_LOAD, dest=0x2000, dims=16),
                    Instruction(Op.EXEC_MAC, dims=dimensions(4, 3, 2), flags=3),
                    Instruction(Op.POST_PROC, dest=0x2040, dims=dimensions(4, 3), flags=5),
                    Instruction(Op.FORWARD, dest=1, dims=dimensions(4, 3)),
                    Instruction(Op.STREAM_STORE, src=0x2040, dims=16),
                    Instruction(Op.LOOP_START, src=4, dest=0x3910),
                    Instruction(Op.LOOP_END, dest=0x3910),
                    Instruction(Op.LOAD_OPERAND, src=0x3000, dest=2, dims=16),
                    Instruction(Op.HALT)]
        for ins in examples:
            self.assertEqual(Instruction.decode(ins.encode()), ins)
        self.assertEqual({i.op for i in examples}, set(Op))

    def test_bad_instructions(self):
        cases = [Instruction(Op.LOAD_OPERAND, src=0x2001, dims=16),
                 Instruction(Op.LOAD_OPERAND, src=0x2000, dest=1, dims=16),
                 Instruction(Op.EXEC_MAC, dims=dimensions(5, 4, 4)),
                 Instruction(Op.EXEC_MAC, dims=dimensions(4, 4, 0)),
                 Instruction(Op.HALT, flags=1),
                 Instruction(Op.STREAM_LOAD, dest=0x2000, dims=32)]
        for ins in cases:
            with self.subTest(ins=ins), self.assertRaises(ValueError):
                ins.encode()
        with self.assertRaises(ValueError):
            Instruction.decode(bytes(16))
        with self.assertRaises(ValueError):
            disassemble(bytes(17))

    def test_single_loop(self):
        code = [Instruction(Op.LOOP_START, src=3, dest=0x3910),
                Instruction(Op.STREAM_LOAD, dest=0x2000, dims=16),
                Instruction(Op.STREAM_STORE, src=0x2000, dims=16),
                Instruction(Op.LOOP_END, dest=0x3910), Instruction(Op.HALT)]
        data = bytes(range(48))
        result = execute(program_image(code), data)
        self.assertEqual(result["output"], data)
        self.assertEqual(result["instructions"], 11)


class NumericTests(unittest.TestCase):
    def test_signed_rounding_and_clamp(self):
        for shift in range(0, 12):
            for value in range(-200, 201):
                self.assertEqual(round_shift(value, shift), round(Fraction(value, 1 << shift)))
        self.assertEqual(requantize(-1000, 1, 0, False), -128)
        self.assertEqual(requantize(1000, 1, 0, False), 127)
        self.assertEqual(requantize(-1000, 1, 0, True), 0)

    def test_quantization_bias_units_zero_and_overflow(self):
        zero = Layer("zero", np.zeros((4, 4)), np.zeros(4))
        q = quantize([zero], np.zeros((5, 4)))[0]
        self.assertEqual(q["input_scale"], 1)
        self.assertEqual(q["bias"], [0]*4)
        bad = Layer("bad", np.ones((4, 4)), np.full(4, 1e20))
        with self.assertRaisesRegex(ValueError, "INT32"):
            quantize([bad], np.ones((4, 4)))
        with self.assertRaises(ValueError):
            quantize([zero], np.full((4, 4), np.nan))

    def test_tensor_layout(self):
        x = np.arange(21, dtype=np.int8).reshape(3, 7)
        beats = pack_tensor(x)
        self.assertEqual(len(beats), 32)
        self.assertEqual(list(beats[:8]), [0,1,2,3,7,8,9,10])
        self.assertEqual(list(beats[16:20]), [4,5,6,0])
        np.testing.assert_array_equal(unpack_tensor(beats, 3, 7), x)


class FrontendTests(unittest.TestCase):
    def test_example_onnx_against_onnx_reference(self):
        path = EXAMPLES / "model.onnx"
        layers = load_model(path)
        x = np.load(EXAMPLES / "inputs.npy")
        y = ReferenceEvaluator(str(path)).run(None, {"input": x})[0]
        np.testing.assert_allclose(run_float(layers, x)[-1], y, rtol=1e-5, atol=1e-6)
        j = load_model(EXAMPLES / "model.json")
        for a, b in zip(layers, j):
            np.testing.assert_array_equal(a.weights, b.weights)
            np.testing.assert_array_equal(a.bias, b.bias)

    def test_matmul_add_and_gemm_attributes(self):
        rng = np.random.default_rng(123)
        w, b = rng.normal(size=(3, 5)).astype(np.float32), rng.normal(size=(5,)).astype(np.float32)
        for kind in ("matmul", "gemm"):
            if kind == "matmul":
                nodes = [helper.make_node("MatMul", ["x", "w"], ["z"]),
                         helper.make_node("Add", ["b", "z"], ["a"]),
                         helper.make_node("Relu", ["a"], ["y"])]
            else:
                nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["a"], alpha=0.75, beta=1.25),
                         helper.make_node("Relu", ["a"], ["y"])]
            graph = helper.make_graph(nodes, kind,
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 5])],
                [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)], ir_version=9)
            with tempfile.TemporaryDirectory() as t:
                path = Path(t)/"test.onnx"
                onnx.save(model, path)
                x = rng.normal(size=(4, 3)).astype(np.float32)
                y = ReferenceEvaluator(model).run(None, {"x": x})[0]
                np.testing.assert_allclose(run_float(load_model(path), x)[-1], y, rtol=1e-5, atol=1e-6)

    def test_unsupported_operator_rejected(self):
        model = onnx.load(EXAMPLES / "model.onnx")
        model.graph.node[1].op_type = "Sigmoid"
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/"bad.onnx"
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_model(path)


class EndToEndTests(unittest.TestCase):
    def test_tiling_all_batches_forward_modes_and_partial_batches(self):
        rng = np.random.default_rng(1234)
        shapes = [(4,4,4), (7,5,3), (5,9,2), (4,3,7), (1,1), (8,8), (9,3,2)]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for widths in shapes:
                layers = [Layer(f"L{i}", rng.normal(0, .4, (n,k)), rng.normal(0,.1,n),
                                "relu" if i < len(widths)-2 else "none")
                          for i, (k,n) in enumerate(zip(widths, widths[1:]))]
                save_json(root/"m.json", layers)
                np.save(root/"cal.npy", rng.normal(0,2,(32,widths[0])))
                x = rng.integers(-128,128,(9,widths[0]),dtype=np.int16)
                x[0] = -128
                x[1] = 127
                for batch in range(1,5):
                    for forward in (False, True):
                        with self.subTest(widths=widths,batch=batch,forward=forward):
                            m = compile_model(root/"m.json",root/"cal.npy",root/"out",batch,forward)
                            result = run_directory(root/"out", x, quantized=True)
                            self.assertEqual(len(result["output_int8"]), 9)
                            self.assertEqual(result["runs"][0]["useful_macs"], m["useful_macs_per_inference"])
                            data = (root/"out/sram.bin").read_bytes()
                            hexdata = b"".join(int(line,16).to_bytes(16,"little")
                                              for line in (root/"out/sram.hex").read_text().splitlines())
                            self.assertEqual(data, hexdata)

    def test_deterministic_build(self):
        layers = load_model(EXAMPLES/"model.onnx")
        cal = np.load(EXAMPLES/"calibration.npy")
        self.assertEqual(build(layers, cal), build(layers, cal))

    def test_program_capacity(self):
        layers = [Layer(f"L{i}", np.ones((8,8)), np.zeros(8)) for i in range(8)]
        with self.assertRaisesRegex(ValueError, "program capacity"):
            build(layers, np.ones((8,8)), forward=False)

    def test_parameter_capacity(self):
        layer = Layer("big", np.ones((128,128)), np.zeros(128))
        with self.assertRaisesRegex(ValueError, "weights capacity"):
            build([layer], np.ones((2,128)))

    def test_binary_faults(self):
        layers = load_model(EXAMPLES/"model.onnx")
        image, program, manifest, _, _ = build(layers, np.load(EXAMPLES/"calibration.npy"))
        inputs = bytes(16)
        bad = bytearray(image)
        bad[ENTRY_PC+15] &= 0x0f  # opcode zero
        with self.assertRaises(ValueError):
            execute(bad, inputs)
        bad = bytearray(image)
        cfg = manifest["layers"][0]["config"]
        bad[cfg:cfg+4] = bytes(4)  # multiplier zero
        with self.assertRaisesRegex(ValueError, "configuration"):
            execute(bad, inputs)
        bad = bytearray(image)
        bias = manifest["layers"][0]["bias"][0]
        bad[bias:bias+16] = struct.pack("<iiii", *([2**31-1]*4))
        weights = manifest["layers"][0]["weights"][0][0]
        bad[weights:weights+16] = bytes([127]*16)
        with self.assertRaisesRegex(ValueError, "overflow"):
            execute(bad, bytes([127]*16))
        invalid = program_image([Instruction(Op.EXEC_MAC,dims=dimensions(4,4,4)),Instruction(Op.HALT)])
        with self.assertRaisesRegex(ValueError, "invalid operands"):
            execute(invalid, b"")
        with self.assertRaisesRegex(ValueError, "underflow"):
            execute(image,b"")
        with self.assertRaisesRegex(ValueError, "unused"):
            execute(image,bytes(32))


if __name__ == "__main__":
    unittest.main()
