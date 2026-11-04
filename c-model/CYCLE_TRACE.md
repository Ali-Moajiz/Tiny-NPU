# Static pipeline cycle trace

Numbers describe modeled clock edges. The default build uses PREFETCH=1, FUSE=1 and POST_LATENCY=1.

| Edge | Read issue | Capture or compute |
|---:|---|---|
| 1 | W1 → bank 0 | |
| 2 | Input → bank 0 | Capture W1 |
| 3 | B1 → bank 0 | Capture input |
| 4 | CFG1 → bank 0 | Capture B1 |
| 5 | Prefetch W2 → bank 1 | Capture CFG1 and compute L1 k=0 |
| 6 | Prefetch B2 → bank 1 | Capture W2 and compute L1 k=1 |
| 7 | Prefetch CFG2 → bank 1 | Capture B2 and compute L1 k=2 |
| 8 | | Capture CFG2 and compute L1 k=3 |
| 9 | | Fused L1 post |
| 10 | | Forward hidden tile into input bank 1 |
| 11–14 | | Compute L2 k=0..3 |
| 15 | | Fused L2 post |
| 16 | | Store output to SRAM line 0x200 |

With POST_LATENCY=0, edge 9 forwards the result directly, L2 computes at edges 10–13, and edge 14 performs fused post plus store.

## Measured before and after

Each row applies to BATCH=1 and BATCH=4. Utilization triples are **baseline / latency 1 / latency 0**. Compute utilization is 25% for batch 1 and 100% for batch 4 in every row. SRAM traffic is listed as reads/writes and applies to all three designs.

| PREFETCH | FUSE | Edges before / L1 / L0 | SRAM reads/writes | BATCH=1 whole run (%) | BATCH=4 whole run (%) |
|---:|---:|---|---|---|---|
| 0 | 0 | 26 / 20 / 18 | 8/2 | 7.69 / 10.00 / 11.11 | 30.77 / 40.00 / 44.44 |
| 0 | 1 | 24 / 19 / 17 | 7/1 | 8.33 / 10.53 / 11.76 | 33.33 / 42.11 / 47.06 |
| 1 | 0 | 23 / 18 / 16 | 8/2 | 8.70 / 11.11 / 12.50 | 34.78 / 44.44 / 50.00 |
| 1 | 1 | 20 / 16 / 14 | 7/1 | 10.00 / 12.50 / 14.29 | 40.00 / 50.00 / 57.14 |

The tested cycle formula is 4 cold read issues + 8 MAC edges + 2 post edges + 2 × POST_LATENCY. Add two edges for an unfused reload when prefetched. Without prefetch, add three edges for layer-2 parameter issue and one more if the hidden tile also needs an SRAM read.

## Approved interpretations and deviations

- CFG1 captures on edge 5 while L1 consumes W/I/B. The lifetime check is per file and bank.
- The dependent two-layer trace demonstrates compute + prefetch and, at latency 0, post + store. These are separate overlaps, as approved.
- Spatial-K remains an estimate of 9–10 edges with 64 INT8 multipliers and a four-product reduction tree per PE, under the same read timing. The implemented workload has only layer and k dimensions. A schedule index and a one-edge read tag are sequencer state.
- There were no further deviations from D1–D8. Latency is measured in simulated edges.
