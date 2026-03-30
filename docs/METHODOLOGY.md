# Methodology and Mathematical Specification

This document gives a formal, end-to-end description of the **Head Trajectories** project: the scientific question, the experimental pipeline, the probe dataset, the five head-behavior scores, threshold calibration, classification, and the trajectory analyses derived from those labels.

It is the mathematical companion to:

- [README.md](../README.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 1. Scientific Objective

Most interpretability work identifies attention-head behaviors **after training**. This project asks a developmental question instead:

> **When, and in what order, do transformer attention heads acquire recognizable behavioral roles during training?**

The project operationalizes this question by:

1. Training a transformer from random initialization.
2. Saving checkpoints throughout training, especially densely early on.
3. Running a fixed held-out probe dataset through every checkpoint.
4. Scoring every attention head on five interpretable behaviors.
5. Classifying each head at each checkpoint.
6. Tracking each head’s label sequence over time.

The resulting object is a **trajectory** for each head:

$$
(\text{label at step } t_1,\ \text{label at step } t_2,\ \dots,\ \text{label at step } t_K).
$$

---

## 2. Experimental Pipeline

At a high level, the pipeline is:

$$
\text{train model} \rightarrow \text{save checkpoints} \rightarrow \text{extract attention maps} \rightarrow \text{score heads} \rightarrow \text{classify heads} \rightarrow \text{analyze trajectories}.
$$

More concretely:

### Training phase

- A decoder-only transformer is trained from scratch according to a named experiment profile.
- The current repository supports notebook-scale comparison profiles on WikiText-103 and LM1B, plus longer OpenWebText runs.
- Checkpoints are saved on a **dense-early, sparse-late** schedule.

### Probing phase

- A fixed held-out probe dataset is run through each checkpoint.
- Raw attention maps are extracted for every layer and head.
- Each head receives five scalar scores.

### Classification phase

- Scores are compared against calibrated thresholds.
- The full raw score tensor is preserved in results files; the categorical label is a summary layer rather than the only retained representation.
- Each head is assigned one of six labels:

$$
\mathcal{Y} = \{\texttt{UNDIFFERENTIATED},\ \texttt{SINK},\ \texttt{PREV\_TOKEN},\ \texttt{INDUCTION},\ \texttt{POSITIONAL},\ \texttt{SEMANTIC}\}.
$$

### Analysis phase

- Compute global type-fraction curves over training.
- Compute per-layer curves.
- Extract per-head trajectories.
- Measure onset, stability, transitions, and phase-like behavior.

---

## 3. Core Objects and Notation

We use the following notation.

### 3.1 Model and checkpoints

- Let $ \theta^{(k)} $ denote model parameters at checkpoint $k$.
- Let the model have:
  - $L$ layers
  - $H$ attention heads per layer
  - sequence length $T$

### 3.2 Attention maps

For checkpoint $k$, layer $\ell$, head $h$, and probe sequence $n$, let

$$
A^{(k,\ell,h)}_n \in [0,1]^{T \times T}
$$

be the post-softmax attention matrix for that head on that sequence.

Because the model is causal:

$$
A^{(k,\ell,h)}_n[t,j] = 0 \quad \text{for } j > t,
$$

and each row is row-stochastic:

$$
\sum_{j=0}^{T-1} A^{(k,\ell,h)}_n[t,j] = 1.
$$

### 3.3 Probe partitions

The probe dataset is partitioned into:

- **general probes**
- **induction probes**
- **positional probes**

Each partition is used for a specific subset of scores.

---

## 4. Probe Dataset

The probe dataset is a **fixed, immutable, held-out dataset** used only for measurement. It is built once and then reused for every checkpoint and every run.

This immutability is essential: it ensures that changes in scores over training reflect changes in the model, not changes in the measurement instrument.

### 4.1 General probes

General probes are ordinary held-out token sequences:

$$
X^{\text{gen}} = \{x^{\text{gen}}_1, \dots, x^{\text{gen}}_{N_g}\}, \qquad x^{\text{gen}}_n \in \{0,\dots,V-1\}^T.
$$

They are used for:

- sink score
- previous-token score
- semantic score

### 4.2 Induction probes

Induction probes are held-out sequences modified to contain a repeated subsequence.

For each induction probe $n$, the dataset stores:

- a token sequence $x^{\text{ind}}_n$
- $p^{(1)}_n$: start index of the first occurrence
- $p^{(2)}_n$: start index of the second occurrence

The construction is designed so that a genuine induction head can attend from the second occurrence to what followed the first occurrence.

### 4.3 Positional probes

Positional probes are stored as content-distinct sequence pairs of equal length:

$$
\bigl(x^{(a)}_i,\ x^{(b)}_i\bigr), \qquad i=1,\dots,N_p.
$$

These are used to test whether a head’s attention pattern is driven by **position** rather than **content**.

---

## 5. Five Head-Behavior Scores

For a fixed checkpoint $k$, layer $\ell$, and head $h$, define the score vector

$$
s^{(k,\ell,h)} =
\bigl(
s_{\text{sink}},
s_{\text{prev}},
s_{\text{ind}},
s_{\text{pos}},
s_{\text{sem}}
\bigr).
$$

Below, we suppress $(k,\ell,h)$ in the notation for readability.

---

### 5.1 Sink score

**Intuition:** a sink head anchors attention to a fixed absolute position (e.g., token 0) regardless of query position or sequence content. This is distinct from sharpness: a previous-token head is sharp but its argmax slides with $t$, so it does not score high on this metric.

For general probes $A^{\text{gen}}_n$, define:

$$
s_{\text{sink}}
=
\max_{j=0}^{T-1}
\left(
\frac{1}{N_g}
\sum_{n=1}^{N_g}
\frac{\sum_{t=0}^{T-1} A^{\text{gen}}_n[t,j]}{T - j}
\right).
$$

The normalization by $(T - j)$ accounts for causal mask geometry: key position $j$ is reachable from exactly $(T - j)$ query positions. We divide by this count to get the mean attention per reachable query, then average across sequences and take the maximum over key positions.

Interpretation:

- A true sink head routes all queries to the same fixed position, yielding a score near 1.0.
- A previous-token head distributes attention across all key positions (each query attends to a different key), yielding a score around 0.5.
- Uniform causal attention yields a score around $1/T$.

**Note:** Random unconstrained softmax attention can occasionally produce scores slightly above 1.0 due to variance, but this is rare in practice with learned causal attention patterns.

---

### 5.2 Previous-token score

**Intuition:** a previous-token head attends from position $t$ to position $t-1$.

Using general probes:

$$
s_{\text{prev}}
=
\frac{1}{N_g (T-1)}
\sum_{n=1}^{N_g}\sum_{t=1}^{T-1}
A^{\text{gen}}_n[t,t-1].
$$

Interpretation:

- Large values indicate systematic one-step-back attention.

---

### 5.3 Induction score

**Intuition:** an induction head detects a repeated pattern and attends to what came after the earlier occurrence.

For induction probe $n$, let:

- $p^{(1)}_n$ be the start of the first occurrence
- $p^{(2)}_n$ be the start of the second occurrence

Then the score reads the attention weight:

$$
A^{\text{ind}}_n\bigl[p^{(2)}_n,\ p^{(1)}_n + 1\bigr].
$$

The overall induction score is:

$$
s_{\text{ind}}
=
\frac{1}{N_i}
\sum_{n=1}^{N_i}
A^{\text{ind}}_n\bigl[p^{(2)}_n,\ p^{(1)}_n + 1\bigr].
$$

Interpretation:

- High score means that when the repeated pattern begins again, the head attends to the token that followed the first occurrence, which is the canonical induction-head signature.

---

### 5.4 Positional score

**Intuition:** a positional head should produce similar attention patterns on two different sequences if they have the same length, because its behavior depends mainly on position, not content.

Let the positional probe pairs be:

$$
\{(a_i, b_i)\}_{i=1}^{N_p}.
$$

For pair $i$, let $A_i$ and $B_i$ denote the corresponding attention matrices.

The implementation smooths $B_i$ row-wise with $\varepsilon = 10^{-9}$ and renormalizes:

$$
\widetilde{B}_i[t,j]
=
\frac{B_i[t,j] + \varepsilon}{\sum_{u=0}^{T-1}(B_i[t,u] + \varepsilon)}.
$$

For each row $t$, define the KL divergence:

$$
D_{\mathrm{KL}}\!\left(A_i[t,\cdot] \,\|\, \widetilde{B}_i[t,\cdot]\right)
=
\sum_{j=0}^{T-1}
A_i[t,j]
\log\frac{A_i[t,j]}{\widetilde{B}_i[t,j]}.
$$

Average over rows:

$$
\overline{D}_i
=
\frac{1}{T}\sum_{t=0}^{T-1}
D_{\mathrm{KL}}\!\left(A_i[t,\cdot] \,\|\, \widetilde{B}_i[t,\cdot]\right).
$$

Convert divergence to a similarity-like score:

$$
s^{(i)}_{\text{pos}} = \max(0,\ 1 - \overline{D}_i).
$$

Then average across pairs:

$$
s_{\text{pos}}
=
\frac{1}{N_p}\sum_{i=1}^{N_p} s^{(i)}_{\text{pos}}.
$$

Interpretation:

- High score means “attention pattern is nearly the same across different content.”
- Low score means “attention pattern changes substantially with content.”

---

### 5.5 Semantic score

**Intuition:** a semantic head should attend to tokens whose meanings are similar to the query token.

Let $E \in \mathbb{R}^{V \times d}$ be the model’s **current checkpoint embedding matrix**, and define row-normalized embeddings:

$$
\widehat{E}_v = \frac{E_v}{\lVert E_v \rVert_2}.
$$

For general probe sequence $n$, let token IDs be:

$$
x_n = (x_{n,0}, \dots, x_{n,T-1}).
$$

For each position $t \ge 4$, define:

- the causal attention vector

$$
a_{n,t} =
\bigl(
A^{\text{gen}}_n[t,0],\dots,A^{\text{gen}}_n[t,t]
\bigr),
$$

- the causal semantic-similarity vector

$$
c_{n,t} =
\bigl(
\cos(\widehat{E}_{x_{n,t}}, \widehat{E}_{x_{n,0}}),\dots,
\cos(\widehat{E}_{x_{n,t}}, \widehat{E}_{x_{n,t}})
\bigr).
$$

**Exclusion masking:** Before computing correlation, we remove three confounded positions:
- $j = t$ (identity position, cosine similarity always 1.0)
- $j = t-1$ (previous-token position, confounds with prev-token heads)
- $j = 0$ (sink position, confounds with sink heads)

Let $M_{n,t}$ be the mask that excludes these positions. We require $|M_{n,t}| \ge 6$ (minimum 6 valid points for stable Pearson correlation). If fewer than 6 points remain after masking, position $t$ is skipped.

Then compute the Pearson correlation on the masked vectors:

$$
\rho(a_{n,t} \odot M_{n,t}, c_{n,t} \odot M_{n,t})
=
\frac{
\mathbb{E}\!\left[(a_{n,t} - \bar a_{n,t})(c_{n,t} - \bar c_{n,t})\right]
}{
\sigma(a_{n,t})\,\sigma(c_{n,t})
},
$$

for all $(n,t)$ where both standard deviations are nonzero, the correlation is finite, and $|M_{n,t}| \ge 6$.

The semantic score is the empirical mean over valid masked positions:

$$
s_{\text{sem}}
=
\mathbb{E}_{(n,t)\in\mathcal{V}}
\bigl[\rho(a_{n,t} \odot M_{n,t}, c_{n,t} \odot M_{n,t})\bigr],
$$

where $\mathcal{V}$ is the set of valid sequence-position pairs with sufficient unmasked points.

Interpretation:

- Positive values mean attention aligns with semantic similarity (after removing structural confounds).
- Near-zero values mean little semantic alignment.
- Negative values mean the head anti-correlates with semantic similarity.

---

## 6. Threshold Calibration

Raw scores across the five metrics are **not directly comparable**. A score of $0.10$ may be highly meaningful for one metric and unremarkable for another.

The project therefore calibrates a threshold vector

$$
\tau = (\tau_{\text{sink}}, \tau_{\text{prev}}, \tau_{\text{ind}}, \tau_{\text{pos}}, \tau_{\text{sem}})
$$

from a random baseline.

### 6.1 Why calibration is needed

The goal is to answer:

> How large must a score be before we stop treating it as random-noise behavior?

### 6.2 Calibration procedure

For a given architecture:

1. Initialize a **random model** with that architecture.
2. Run the fixed probe dataset through it.
3. Extract all attention maps.
4. Scramble key positions independently within each valid causal row to destroy fixed-key anchoring and other structured patterns while preserving causal support and row-stochasticity.
5. Score every head on all five metrics.

This null matters for the sink metric in particular: row-shuffling leaves fixed-key aggregation largely unchanged, so the calibration baseline must scramble **keys within rows**, not rows themselves.

If $r_{u,m}$ is the random-baseline score of head $u$ on metric $m$, define:

$$
\mu_m = \frac{1}{H_{\text{tot}}}\sum_{u=1}^{H_{\text{tot}}} r_{u,m},
$$

$$
\sigma_m = \sqrt{
\frac{1}{H_{\text{tot}}}
\sum_{u=1}^{H_{\text{tot}}}(r_{u,m} - \mu_m)^2
},
$$

where

$$
H_{\text{tot}} = L \times H
$$

is the total number of heads in the model.

Then the single-seed threshold is:

$$
\tau_m =
\begin{cases}
\mu_m + 2\sigma_m, & m \in \{\text{sink}, \text{prev-token}, \text{induction}, \text{positional}\}, \\
Q_{0.99}(r_{\cdot,m}), & m = \text{semantic},
\end{cases}
$$

where \(Q_{0.99}(r_{\cdot,m})\) is the empirical 99th percentile of the semantic null scores across heads for that calibration seed.

The semantic exception is intentional. The semantic metric is a signed Pearson-correlation statistic, and its null variance collapses sharply once per-position correlations are averaged into a final per-head score. In practice, using \(\mu_m + 2\sigma_m\) for semantic produced thresholds that were too permissive. A high null quantile is therefore used for semantic while the other four metrics retain the original mean-plus-two-standard-deviations rule.

### 6.3 Multi-seed calibration

The repository repeats this calibration across several random seeds and stores:

- per-seed threshold vectors
- per-seed metric means
- per-seed metric standard deviations
- per-seed metric quantiles (`p95`, `p99`)
- whether any threshold was non-positive and would require defensive sanitization at classification time
- their empirical mean
- their empirical standard deviation

The final threshold vector used in the main pipeline is the mean threshold across calibration seeds.

---

## 7. Head Classification Rule

Given a real checkpoint score vector

$$
s = (s_1,\dots,s_5)
$$

and threshold vector

$$
\tau = (\tau_1,\dots,\tau_5),
$$

define normalized scores:

$$
z_m = \frac{s_m}{\tau_m}.
$$

In implementation, the classifier validates thresholds before normalization. Non-finite thresholds are rejected. Non-positive thresholds are preserved as raw calibration outputs for reporting, but are replaced by a small positive floor **only for safe division**. The below-threshold check still uses the raw calibrated thresholds.

The classification rule is:

### Case 1: all scores below threshold

If

$$
s_m < \tau_m \quad \forall m,
$$

then the head is classified as

$$
\texttt{UNDIFFERENTIATED}.
$$

### Case 2: ambiguous winner

Let $z_{(1)} \ge z_{(2)} \ge \dots $ be the sorted normalized scores.

If

$$
z_{(1)} - z_{(2)} < \delta,
$$

with tie tolerance

$$
\delta = 0.05,
$$

then the head is also classified as

$$
\texttt{UNDIFFERENTIATED},
$$

and the tie is logged.

### Case 3: clear winner

Otherwise, assign the label corresponding to:

$$
\arg\max_m z_m.
$$

So the classifier is:

$$
\text{label}(s)
=
\begin{cases}
\texttt{UNDIFFERENTIATED}, & s_m < \tau_m \ \forall m, \\
\texttt{UNDIFFERENTIATED}, & z_{(1)} - z_{(2)} < \delta, \\
\arg\max_m z_m, & \text{otherwise.}
\end{cases}
$$

---

## 8. Trajectories

Once every head is classified at every checkpoint, the central object of the project is the label trajectory:

$$
y^{(\ell,h)} =
\bigl(
y^{(\ell,h)}_1,\ y^{(\ell,h)}_2,\ \dots,\ y^{(\ell,h)}_K
\bigr).
$$

This supports several levels of analysis.

### 8.1 Global curves

For type $c \in \mathcal{Y}$, define the fraction of heads of type $c$ at checkpoint $k$:

$$
f_{k,c}
=
\frac{1}{LH}
\sum_{\ell=1}^{L}\sum_{h=1}^{H}
\mathbf{1}\{y^{(\ell,h)}_k = c\}.
$$

Across multiple seeds, these are averaged to yield mean and standard deviation bands.

### 8.2 Per-layer curves

For layer $\ell$:

$$
f_{k,c}^{(\ell)}
=
\frac{1}{H}
\sum_{h=1}^{H}
\mathbf{1}\{y^{(\ell,h)}_k = c\}.
$$

This tests whether lower layers specialize earlier than higher layers.

### 8.3 Head-level transitions

For each head, count type changes:

$$
\Delta^{(\ell,h)}
=
\sum_{k=2}^{K}
\mathbf{1}\{y^{(\ell,h)}_k \ne y^{(\ell,h)}_{k-1}\}.
$$

This quantifies stability versus re-specialization.

### 8.4 Mixed behavior and label compression

The dominant-label view is intentionally lossy. A head can exceed threshold for multiple behaviors at once, yet still receive a single categorical label because the classifier chooses the largest threshold-normalized score when there is a clear winner.

This means:

- `UNDIFFERENTIATED` includes both weak heads and ambiguous near-ties
- a head labeled `PREV_TOKEN` can still have substantial `SINK` or `SEMANTIC` signal
- dominant labels are useful for visualization, but raw scores and threshold-presence analyses are often needed to interpret behavioral overlap

In current comparison runs, this distinction is empirically important: many heads exceed thresholds for multiple behaviors simultaneously even when one dominant label accounts for the final trajectory summary.

---

## 9. Hypothesis Tests

The project’s five hypotheses are operationalized through these trajectory objects.

### H1. Sink-first among learned types

Canonical wording:

> **Among learned head types, sink onset should occur no later than prev-token, induction, or semantic onset.**

Operationally:

- compute learned onset steps using `exclude_positional_init=True`
- require `SINK` onset to be defined
- test

$$
t_{\mathrm{sink}} \le t_{\mathrm{prev}}, \qquad
t_{\mathrm{sink}} \le t_{\mathrm{ind}}, \qquad
t_{\mathrm{sink}} \le t_{\mathrm{sem}},
$$

where undefined onset for another type is treated as “not earlier than sink”

Current status: this is an evaluated hypothesis, not a settled result. Recent single-seed comparison runs do not support a strong sink-first claim.

### H2. Ordered development

Canonical wording:

> **After separating architectural positional structure at initialization from learned specialization, the learned onset order should be**
>
> $$
> \text{sink} \le \text{prev\_token} < \text{induction} < \text{semantic}.
> $$

**Note on positional onset:** Positional heads can appear at step 0 due to Rotary Position Embeddings (RoPE), which create deterministic position-based attention patterns before any learning occurs. This is an architectural feature, not learned specialization. The hypothesis therefore distinguishes:
- **Architectural positional**: RoPE-driven structure at initialization (step 0)
- **Learned specialization**: Sink, prev-token, induction, and semantic behaviors that emerge during training

Operationally:

- report architectural positional onset separately from learned onset
- compute learned onset steps with `exclude_positional_init=True`
- require all four learned types to have defined onset
- test

$$
t_{\mathrm{sink}} \le t_{\mathrm{prev}} < t_{\mathrm{ind}} < t_{\mathrm{sem}}.
$$

Current status: this ordering is best treated as a hypothesis under test. Recent single-seed comparison runs do not robustly support it.

### H3. Layer stratification

Canonical wording:

> **Lower layers should reach substantial specialization earlier than higher layers.**

Operationally:

- compute non-undifferentiated fraction per layer over time
- define layer onset as the first checkpoint where the layer reaches at least 50% specialized heads
- test whether layer onset is non-decreasing with layer depth

Current status: this remains open. Some runs show lower-layer activity earlier, but stronger multi-seed evidence is still needed before making a strong claim.

### H4. Phase transition

Canonical wording:

> **Induction heads may emerge through an abrupt phase-like transition rather than a smooth gradual rise.**

Operationally:

- inspect induction counts as a function of step
- compute crossings at 10%, 25%, and 50% of final induction count
- measure discontinuity score from the induction-count slope profile
- check for a nearby validation-loss inflection around the induction crossing

Current status: this remains open. Weak or rare induction in the current comparison runs limits how strong a phase-transition claim can be.

### H5. Sink persistence

Canonical wording:

> **Once a head becomes a sink, it should usually remain a sink for most of its subsequent checkpoints.**

Operationally:

- count transitions out of sink state
- compute sink persistence for every head ever labeled `SINK`
- compare sink stability with overall type-change behavior

Current status: still under evaluation. Because dominant `SINK` labels are rare in current comparison runs, persistence evidence is not yet strong.

---

## 10. Why the Method Is Structured This Way

The methodology is built around three principles.

### 10.1 Same probe set for every checkpoint

This makes the measurement instrument fixed across time.

### 10.2 Random-baseline calibration

This prevents arbitrary raw-score comparisons across behavior types.

### 10.3 Checkpoint trajectories, not just final labels

The goal is developmental structure, so the meaningful object is not a final snapshot but the entire path:

$$
\theta^{(1)} \rightarrow \theta^{(2)} \rightarrow \cdots \rightarrow \theta^{(K)}.
$$

---

## 11. Practical Interpretation

When reading results:

- A high score means a head exhibits a behavior strongly on the fixed probe set.
- A high **normalized** score means that behavior is strong relative to its random baseline.
- A head can remain undifferentiated either because it is genuinely weak or because multiple behaviors are too close to separate cleanly.
- A dominant label should be read as the strongest currently expressed behavior, not the head's full identity.
- Onset times are operational definitions, not metaphysical truths.

This project therefore measures:

> **behavioral specialization relative to a calibrated random baseline, tracked continuously across training.**

---

## 12. Code Mapping

For readers moving between the math and the implementation:

- Probe construction: [data/probe.py](../data/probe.py)
- Threshold calibration: [data/calibration.py](../data/calibration.py)
- Score definitions: [probing/scores.py](../probing/scores.py)
- Classification: [probing/classifier.py](../probing/classifier.py)
- Checkpoint extraction: [probing/extractor.py](../probing/extractor.py)
- Probing pipeline: [probing/pipeline.py](../probing/pipeline.py)
- Trajectory analysis: [analysis/trajectories.py](../analysis/trajectories.py)

---

## 13. Summary

In one sentence:

> The project trains a transformer from scratch, probes every attention head at every checkpoint with a fixed held-out dataset, scores heads on five interpretable behaviors, calibrates significance using random-baseline thresholds, and then studies the resulting label trajectories over training.

That is the full methodology in its most compact form.
