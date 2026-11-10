PYTHON ?= python3
BATCH ?= 4
OUT ?= build/example

.PHONY: demo demo-quantized example compile run disasm test binary
demo:
	$(PYTHON) examples/make_example.py
	$(PYTHON) -m tiny_npu compile --model examples/model.onnx --calibration examples/calibration.npy --batch $(BATCH) --out $(OUT)
	$(PYTHON) -m tiny_npu run --image $(OUT) --inputs examples/inputs.npy --out $(OUT)/run.json
example:
	$(PYTHON) examples/make_example.py
compile:
	$(PYTHON) -m tiny_npu compile --model examples/model.onnx --calibration examples/calibration.npy --batch $(BATCH) --out $(OUT)
run:
	$(PYTHON) -m tiny_npu run --image $(OUT) --inputs examples/inputs.npy --out $(OUT)/run.json
disasm:
	$(PYTHON) -m tiny_npu disasm $(OUT)/program.bin
test:
	$(PYTHON) -m unittest discover -s tests -v

demo-quantized:
	$(PYTHON) examples/make_quantized_examples.py
	$(PYTHON) -m tiny_npu examples/qat_qdq.onnx --batch $(BATCH) --out $(OUT)
	$(PYTHON) -m tiny_npu run --image $(OUT) --inputs examples/inputs.npy --out $(OUT)/run.json
binary:
	$(PYTHON) build_binary.py
