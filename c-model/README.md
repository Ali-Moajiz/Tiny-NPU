# Tiny NPU bus model

The 4×4 MAC array, single-port SRAM, and static schedule are modeled as independent blocks. Each block owns one set of registers. Named, typed buses carry signals between block ports, with all endpoints listed in `npu_top.c`. The buses model dedicated wires without arbitration or transactions.

```sh
make -s run BATCH=4
make -s run BATCH=4 POST_LATENCY=0
make -s run TB=1 BATCH=4 LOG=none
make -s check
```

`BATCH=1..4` enables independent vector columns. `PREFETCH` and `FUSE` accept 0 or 1 and default to 1. `POST_LATENCY` defaults to 1. `TB=1` runs the existing 100 random batches with seed 42. `LOG=flow` shows edge events, `LOG=ports` shows typed signals and register updates, and `LOG=both` shows both.

## Add or trace a block

- Define the block's `Ports` and `Regs` beside its implementation. Inputs point to const buses, outputs point to buses driven by the block. Keep register access local to the block during cycle execution.
- Use `BUS_SIGNAL` in `signals.h` to define a typed signal as needed. It wraps the shared `bus_init`, `bus_drive`, and `bus_read` APIs. The bus stores a value copy, so it cannot alias live block registers.
- In `npu_top.c`, instantiate and initialize the block, connect named ports, and add its `BlockOps` to the block list. Evaluate combinational producers before their consumers. Keep the initialized NPU instance at a stable address because its ports reference its buses.
- Implement the applicable callbacks. `publish` drives registered outputs. `eval` drives combinational outputs. `check` verifies all active transfers before any state update. `clock` reads the settled buses and updates local registers in place. The framework rejects duplicate or unauthorized drives and reads of undriven signals. Clock callbacks cannot drive buses and must have their active inputs checked before the clock phase.

For example, SRAM publishes `sram_response` (128 data bits plus valid). All four operand banks read it. The controller's capture enables choose which one updates. A bank's selected output is driven before the clock phase, so a same-edge capture cannot change what the MAC or post-processing reads on that edge.

The callback framework is a C simulation scheduler. Synthesis, timing, and FPGA validation are separate steps.

## Control and datapath

The schedule builder checks the single-port address and access constraints before the first edge. It tracks which operand file and bank has been captured, when its Q becomes available, and whether the same edge consumes that file and bank. Each bank and file receives one load per run. A previous-read control tag drives separate W/I/B/CFG capture enables. SRAM Q is wired in parallel to the register files.

For each layer, four edges issue k=0..3 to the 16 PEs. Bias enters at k=0. The completed accumulator tile feeds one fused stage. Its 16 parallel lanes each use a 32×32→64 multiplier, in addition to the MAC array’s 16 INT8 multipliers. The stage applies optional ReLU, scaling, nearest-even rounding and saturation. At latency 1, a 128-bit output register separates post-processing from destination capture. At latency 0, the SRAM write or input-bank capture shares the post edge and has a longer combinational path. Clock frequency and area require synthesis to quantify.

The scheduled destination is either SRAM or the next input bank. With fusion enabled, the hidden result forwards as INT8. Without fusion, it stores to line 0x201 and reloads. The final result stores to line 0x200 with byte enables for the active vector lanes. Address ranges by type are W 0x000–0x1FF, I 0x200–0x2FF, B 0x300–0x37F and CFG 0x380–0x38F. The model preserves configuration validation and checks INT32 overflow before committing any state.

## Verification

`make check` tests the bus framework and runs both post latencies with UBSan. The framework test covers frozen reads with either clock order and rejection of undriven signals, wrong drivers, and a failed pre-clock check. Each build checks 2,580 vectors against the independent Python oracle, including each neuron and vector at every k. The test asserts formula-derived cycles, SRAM traffic, utilization counters, byte masking, a hand-checked batch-4 edge trace, and injected address, config, overflow, bank and schedule faults.

The 16 CLI builds for BATCH=1/4 across both post latencies and all prefetch/fuse combinations passed with `-Wall -Wextra -Werror`. Both batch-4 latency builds also passed the 100-batch TB under ASan and UBSan. Leak detection was disabled for the sandbox run.

The bundled seven-page draw.io file shows the static hardware schedule and retains the historical pages. The source wiring and framework are documented above. Page 01 shows the active datapath and page 06 shows its schedule. See CYCLE_TRACE.md for the measured comparison and approved scheduling interpretations.
