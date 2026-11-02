# Default schedule: PREFETCH=1, FUSE=1, BATCH=1..4

One traversal models one rising-edge update. Before an edge, all registered buses expose
old Q; the traversal evaluates that edge's work, and the generic scheduler commits Q after
all blocks run. A registered `CAPTURE` logged at E2 is visible to consumers at E3.
Control/demux wires are evaluated in dependency order within the same traversal.

| Edge | SRAM / bank capture work | Compute / downstream work |
|---:|---|---|
| 1 | Read W1 -> bank0 | Cold setup |
| 2 | Capture W1; read input tile -> bank0 | Cold setup |
| 3 | Capture I0; read B1 -> bank0 | Cold setup |
| 4 | Capture B1; read CFG1 -> bank0 | Cold setup |
| 5 | Capture CFG1 | Wait; old CFG0.loaded is still0 |
| 6 | Read W2 -> bank1 | Select bank0; layer1 C1; every active PE seeds bias + W[r,0]*X[v,0] |
| 7 | Capture W2; read B2 -> bank1 | Layer1 C2, feedback own old accumulator |
| 8 | Capture B2; read CFG2 -> bank1 | Layer1 C3 |
| 9 | Capture CFG2 | Layer1 C4; register final INT32 tile |
| 10 | No SRAM transfer | Layer1 activation register (ReLU) |
| 11 | No SRAM transfer | Layer1 requantization register (INT8) |
| 12 | Hidden tile -> I bank1 through forwarding mux | No hidden SRAM access; no compute with just-captured input |
| 13 | No SRAM transfer | Select bank1; layer2 C1 |
| 14 | No SRAM transfer | Layer2 C2 |
| 15 | No SRAM transfer | Layer2 C3 |
| 16 | No SRAM transfer | Layer2 C4; register final INT32 tile |
| 17 | No SRAM transfer | Layer2 activation register (bypass) |
| 18 | No SRAM transfer | Layer2 requantization register (INT8) |
| 19 | No SRAM transfer | FIFO captures final128-bit tile |
| 20 | Masked write to0x200; FIFO pop | DONE after successful full traversal |

The compute-only schedule for neuron1, vector1 with input[1,2,3,4]:

| Compute edge | PE(1,1) | Other active PEs |
|---|---|---|
| L1 C1 / E6 | 42 +19*1 =61 | Same k=0, own neuron weight and independent vector input |
| L1 C2 / E7 | 61 +25*2 =111 | Same k=1, own local accumulator |
| L1 C3 / E8 | 111 -57*3 =-60 | Same k=2, own local accumulator |
| L1 C4 / E9 | -60 +43*4 =112 | Same k=3; all active outputs complete |

This is the SAME PE updating four times. It is NOT MAC11 -> MAC12 -> MAC13 -> MAC14.
For batch4 every one of16 PEs updates at each compute edge. For batch1 only vector-column0
is valid, so four PEs update. The hidden output for vector1 is[1,0,4,2], and final output is
[0,1,-4,-1]. A zero input still counts as a valid multiply; masked vector columns do not.

Read `flow.log` for one vector, `batch4_flow.log` for four distinct vectors, and `ports.log`
for all old/current and next register-bus values. The logs explicitly identify layer, bank,
reduction step, output neuron and vector, rather than reusing the old spatial-column label.
