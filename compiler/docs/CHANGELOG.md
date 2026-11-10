# v0.3.0 — compiler-only PTQ/QAT import update

- New strict ONNX Q/DQ importer: INT8 DQ constants and FLOAT Q/DQ constants,
  shared integer IR and existing ISA v1 backend. Embedded scales are preserved.
- Float ONNX/JSON calibration path remains available. Existing toy example's
  program.bin and sram.bin are byte-identical to v0.2.0 output.
- Existing INT32 biases use accumulator units; FLOAT biases are converted with
  error reporting. M/S approximation is reported. Separate quantized ReLU
  boundaries use explicit identity stages rather than dropping quantization.
- New import_report.json; manifest records import mode and source I/O types.
- Compatible quantized inputs need no calibration file. Incompatible schemes
  are rejected instead of silently recalibrated.
- Compiler executable and build output now reside within compiler/:
  bin/tiny-npu-compiler and build/compiler-binary/ respectively.
- Added Q/DQ examples, source/reference tests, optional real ONNX Runtime export
  test, and quantized-import guide. make demo-quantized runs the new example.
- No changes to C-model, RTL, diagram, ISA encodings, SRAM layout, datapath,
  accumulator arithmetic, or activation/scale block configuration.
