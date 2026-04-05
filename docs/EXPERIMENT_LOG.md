# Experiment Log

This file is a historical log of early pilot runs. For current archived findings and their run reports, see the local `run_exports/` archive rather than treating the notes below as the current state of evidence.

## Current Baseline Conclusions

As of the current default methodology (`enable_natural_induction = false`), the strongest baseline conclusions come from the archived default-methodology WikiText and LM1B runs in `run_exports/`.

Important archive note:

- those archived baseline bundles remain scientifically useful
- but the saved result files were produced before the repo started persisting the full native modern active-set schema
- current code can still load and compare them, but some active-set-style fields are reconstructed through legacy compatibility paths
- therefore they should be trusted most for dominant-label and cross-dataset conclusions, not for over-precise claims about modern saved-schema internals

### Cross-dataset result

The clearest current empirical result is:

- both native current-schema runs are dominated by `AMBIGUOUS` summary labels rather than clean single-role endpoints
- both datasets show near-universal `PREV_TOKEN` activity plus strong `SINK` co-activity
- WikiText retains more `SEMANTIC`, more `INDUCTION`, more overall overlap, and much smaller dominant margins
- LM1B is still mixed, but somewhat cleaner and more concentrated in `PREV_TOKEN > SINK`

In other words, the strongest dataset-sensitive quantity is now not just "which single label wins," but **how mixed the heads remain and which secondary behaviors persist**.

### Classification result

The other important current conclusion is:

- many heads exhibit multiple statistically active behaviors simultaneously
- dominant labels are still useful, but they compress substantial overlap
- the newest native-schema WikiText and LM1B exports make this especially clear: both runs are mostly `AMBIGUOUS` at the dominant-summary level despite strong active behavior structure underneath

This methodology change has now landed: the repository uses an **FDR-based multi-behavior classifier** built on the existing five scores and empirical null calibration.

Current classification/reporting now exposes:

- `WEAK` versus `AMBIGUOUS` non-specialized states
- active behavior set
- dominant behavior plus runner-up behavior
- dominant margin
- number of statistically active behaviors

### Current hypothesis status

- `H1` (sink-first among learned types): not robust in current single-seed comparison runs
- `H2` (ordered development): not robust in current single-seed comparison runs

The codebase now has schema-matched native current-schema seed-42 exports for both:

- `wikitext103_15m_preliminary`
- `lm1b_15m_comparison`

so the mixed-behavior cross-dataset comparison is now on a clean apples-to-apples footing at the single-seed level.

### Next run priority

After the current WikiText and LM1B baselines, the highest-value next work is:

- finish the direct WikiText-vs-LM1B native-schema comparison write-up
- strengthen head-level order analysis
- add synthetic validation and stronger calibration reporting before broadening scope

OpenWebText remains a possible later extension rather than the automatic next step.

## Preliminary Pilot Run: `modal_a100_preliminary`

**Date:** March 2026  
**Goal:** A preliminary run to test the end-to-end pipeline (training, probing, calibration, classification) on a single seed using a real dataset prior to the full OpenWebText run.

### Setup and Scale
- **Dataset:** `Salesforce/wikitext` (`wikitext-103-raw-v1`)
  - Train tokens: ~119M
  - Validation tokens: ~250K
  - Test tokens (used for probes): ~286K
- **Model Config:** 6 Layers, 8 Heads, $d_{\text{model}} = 256$, $d_{\text{ffn}} = 1024$, Block Size = 256
- **Training Config:** 12,000 steps, Cosine LR scheduler, Batch Size = 64
- **Probe Sequences Extracted:** 1,117 available sequences of length 256
- **Evaluated Checkpoints:** 12 checkpoints (dense early: 0, 100, 200, 400, 800, 1500, 2500, 4000, 6000, 8000, 10000, 12000)

### Probe Dataset Integrity
- **General probes:** 240 sequences 
- **Induction probes:** 64 sequences (100% success rate in capturing repeated subsequences)
- **Positional probes:** 32 pairs (64 sequences total)

### Threshold Calibration
Random-baseline thresholds computed over 3 random seeds:
- **Raw pilot thresholds:** `[0.028, 0.0039, 0.0096, 0.000, -0.036]`

*(Note: These raw pilot thresholds exposed a calibration/classification fragility. The main codebase now treats non-positive thresholds as a calibration diagnostic and applies defensive sanitization only for safe normalization.)*

### Training Dynamics
The model learned smoothly over 12,000 steps:
- **Initial Loss:** 10.87 (Step 0)
- **Final/Best Loss:** 3.83 (Step 12,000)

---

## Findings & Trajectory Analysis

### Onset Steps ($\ge$ 5% of heads)
When did each head type cross the 5% threshold (at least 3 heads exhibiting the behavior)?
*   **POSITIONAL:** 0
*   **PREV_TOKEN:** 400
*   **SINK:** 800
*   **UNDIFFERENTIATED, INDUCTION, SEMANTIC:** None (Below 5% threshold)

Note: this pilot section uses the old label vocabulary from that period. In the
current default methodology, `UNDIFFERENTIATED` has been replaced by
`WEAK` and `AMBIGUOUS`.

### Key Insights

**1. The `POSITIONAL` Onset at Step 0 is Architectural, Not Learned**
Heads register as `POSITIONAL` immediately at random initialization. This is a correct architectural feature driven by **Rotary Position Embeddings (RoPE)**. 
* At Step 0, token embeddings are random noise.
* RoPE applies a deterministic, mathematically fixed rotation to the queries and keys purely based on their sequence position.
* Because the token noise washes out, the *only* coherent structural signal driving the attention scores is the RoPE rotation.
* Therefore, the attention map for two completely different sequences (the Positional Probes) look almost identical, yielding a high Positional Score.
* *Note:* The methodology now uses a causal key-scramble null for calibration, but the interpretation remains the same: RoPE can generate immediate positional structure before any learned specialization has occurred.

**2. A Preliminary Developmental Pathway of Sequence Tracking**
By tracing individual head trajectories across the 12 checkpoints, a consistent pathway appears in several heads. Look at these specific heads:
*   `Layer 1, Head 0`: `POS` $\rightarrow$ `SINK` $\rightarrow$ `PREV_TOKEN`
*   `Layer 1, Head 3`: `POS` $\rightarrow$ `SINK` $\rightarrow$ `UNDIFF` $\rightarrow$ `PREV_TOKEN`
*   `Layer 1, Head 6`: `POS` $\rightarrow$ `UNDIFF` $\rightarrow$ `SINK` $\rightarrow$ `PREV` 
*   `Layer 2, Head 2`: `POS` $\rightarrow$ `SINK` $\rightarrow$ `PREV`

**Note on metric evolution:** This pilot used the original SINK metric (measuring sharpness/concentration). The interpretation was: "heads learn to focus sharply before learning WHERE to focus." 

After the pilot, the SINK metric was revised to measure fixed-position anchoring (with causal-mask normalization) rather than sharpness. This change separates true sink heads (anchoring to a fixed position, score ~1.0) from prev-token heads (sliding argmax, score ~0.5). The revised metric provides a mechanistically crisper interpretation: **heads learn fixed-position anchoring before learning dynamic relative tracking.**

The OpenWebText runs use the revised metric, so the SINK → PREV_TOKEN pathway will measure: "fixed anchoring precedes dynamic tracking" rather than "sharpness precedes directional sharpness."

**3. Layer Stratification**
Lower layers and higher layers specialize differently:
*   **Layer 0:** Heads remain highly plastic and mostly default to structural `POSITIONAL` routing, occasionally thrashing into `SEMANTIC` or `SINK` before returning to `POSITIONAL`.
*   **Layer 1 & 2:** These heads cleanly migrate away from `POSITIONAL` via the `SINK` pathway to become stable `PREV_TOKEN` heads.

### Conclusion & Next Steps
The pilot confirms that the pipeline can recover both architectural biases (RoPE at step 0) and plausible learned trajectories (for example `SINK` $\rightarrow$ `PREV_TOKEN`). These results are preliminary rather than conclusive. 
*   *Next Steps:* Scale up to OpenWebText, 15M parameters, and full probe counts to see if `INDUCTION` and `SEMANTIC` heads begin to form over longer trajectories.
