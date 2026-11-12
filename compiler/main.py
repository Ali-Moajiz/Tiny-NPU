#!/usr/bin/env python3
"""START HERE. The complete model-to-binary pipeline is compile_file() below.

    python3 main.py model.onnx --calibration samples.npy
    python3 main.py quantized_model.onnx

The same entry point is packaged as bin/tiny-npu-compiler.
"""
import argparse
from pathlib import Path
import sys

from frontend import load_model
from quantization import prepare_quantization
from compiler import compile_model, write_outputs
from isa import ENTRY_PC, encode_program, disassemble

VERSION = "0.4.0"


def compile_file(model_path, calibration=None, output=None, batch=4, forward=True, debug=False):
    """Five stages, ordinary Python objects, no intermediate files required."""
    # 1. Read and validate the model file.
    model = load_model(model_path)

    # 2. Prepare integer weights, biases, scales and activation settings.
    layers, report = prepare_quantization(model, calibration)

    # 3. Tile matrices, allocate SRAM, and generate typed NPU instructions.
    compilation = compile_model(layers, batch, forward)

    # 4. Encode each instruction as a 128-bit word using our ISA.
    program = encode_program(compilation.instructions)

    # 5. Write program.bin, params.bin and sram.bin. Reports are optional.
    output = Path(output) if output is not None else model.source.with_name(model.source.stem + "_compiled")
    manifest = write_outputs(compilation, program, output, report, model.source, debug)
    return manifest, output


def main(argv=None):
    args = parse_arguments(argv)
    try:
        if args.command == "compile":
            manifest, output = compile_file(args.model, args.calibration, args.out,
                                            args.batch, not args.no_forward, args.debug)
            print(f"Quantization: {manifest['quantization_mode']}")
            print(f"Calibration: {manifest['calibration'] or 'not needed; using embedded scales'}")
            print(f"Compiled {manifest['instruction_count']} instructions ({manifest['program_bytes']} bytes)")
            print(f"ENTRY_PC=0x{manifest['entry_pc']:05X}; useful MAC updates/batch={manifest['useful_macs_per_inference']}")
            print(f"Output: {output.resolve()}")
            print("Files: program.bin, params.bin, sram.bin" + (" + debug reports" if args.debug else ""))
        elif args.command == "disasm":
            print(disassemble(Path(args.program).read_bytes(), args.base), end="")
        elif args.command == "run":
            # Optional verification is separate from compilation.
            import json
            import numpy as np
            from verify import run_directory
            result = run_directory(args.image, np.load(args.inputs, allow_pickle=False), args.quantized)
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2) + "\n")
            print("PASS: binary execution matches the independent integer reference")
            print("INT8 output:", result['output_int8'])
            print(f"Trace: {target.resolve()}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def parse_arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep both "main.py model.onnx" and "main.py compile --model model.onnx".
    if argv and not argv[0].startswith('-') and argv[0] not in ('compile', 'disasm', 'run'):
        argv = ['compile', '--model', argv[0], *argv[1:]]
    parser = argparse.ArgumentParser(
        description="Compile dense/ReLU ONNX models for Tiny NPU (ISA v1).",
        usage="%(prog)s MODEL [--calibration DATA.npy] [--out DIR] [--debug]\n"
              "       %(prog)s {compile,disasm,run} ...")
    parser.add_argument('--version', action='version', version=f'Tiny NPU compiler {VERSION}')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('compile', help='Model -> integer layers -> NPU instructions -> binary files')
    p.add_argument('--model', required=True)
    p.add_argument('--calibration', help='Representative float inputs; omit for compatible Q/DQ ONNX')
    p.add_argument('--out', help='Defaults to MODEL_STEM_compiled/ beside the model')
    p.add_argument('--batch', type=int, choices=range(1, 5), default=4)
    p.add_argument('--no-forward', action='store_true')
    p.add_argument('--debug', action='store_true', help='Also emit assembly, HEX and JSON for inspection/run')
    p = sub.add_parser('disasm', help='Decode instructions from program.bin')
    p.add_argument('program')
    p.add_argument('--base', type=lambda s: int(s, 0), default=ENTRY_PC)
    p = sub.add_parser('run', help='Check a --debug build using the functional binary interpreter')
    p.add_argument('--image', required=True)
    p.add_argument('--inputs', required=True, help='.npy [samples,features], float by default')
    p.add_argument('--quantized', action='store_true', help='Input file already contains raw INT8 codes')
    p.add_argument('--out', default='run.json')
    return parser.parse_args(argv)


if __name__ == '__main__':
    raise SystemExit(main())
