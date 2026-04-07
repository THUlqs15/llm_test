# Energy-Aware Batch Scheduling Experiment Instructions

## Objective

Implement the **Energy-Aware Batch Scheduling Algorithm** (detailed in full below) inside vLLM, replacing its default batch scheduling logic. Then run comparative experiments against vLLM's default scheduler under various parameter configurations to measure the trade-off between **energy consumption** and **SLO (TTFT/TPOT) adherence**. Produce two deliverables: `experiment_results.md` (comparison tables) and `experiment.md` (reproducible workflow + scripts + vLLM modification details).

---

## Environment

- **vLLM source code**: `/home/ubuntu/lqs/vllm` (you may modify any code here)
- **LLM model path**: `/home/ubuntu/lqs/L3`
- **Conda environment**: activate with `conda activate myvllm` before any vLLM-related command
- **Output directory for experiment_results.md and experiment.md**: current working directory (where you are invoked)

---

## Pre-fitted Batch Latency Model Parameters

Use the following fitted parameters from the offline profiling phase. These are **given constants** — do not re-fit them.

| Parameter | Value |
|-----------|-------|
| a_p | 6.974e-02 |
| b_p | 0 (no prefix-cached prefill data) |
| c_p | 3.395e+02 |
| a_d | 2.112e-01 |
| b_d | 4.822e+01 |
| α (alpha) | 0.817 |
| t_c | 57.05 ms |

The per-request processing time model:

```
t_{n,f} = (a_p · ℓ²_{i,n} + b_p · ℓ_{i,n} · ℓ^{kv}_{i,n} + c_p · ℓ_{i,n}) / f    (prefill)
t_{n,f} = (a_d · ℓ^{kv}_{i,n} + b_d · ℓ_{i,n}) / f^α                               (decode)
```

Where:
- `ℓ_{i,n}`: number of tokens of request n processed in iteration i. For decode requests (n ∈ R_i), ℓ_{i,n} = 1. For prefill requests (n ∈ W_i), ℓ_{i,n} = input length of request n (no chunked prefill).
- `ℓ^{kv}_{i,n}`: KV cache length of request n at iteration i.
- `f`: GPU graphics clock frequency in MHz.

The batch execution time:

```
ET_i(B, f) = t_c + Σ_{n ∈ B} t_{n,f}
```

---

## GPU Power Model P(f)

You need to obtain or fit the power consumption model `P(f)` as a function of GPU graphics clock frequency `f`.

**Step 1**: Query supported GPU frequencies:
```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
```

**Step 2**: For each supported frequency, measure GPU power draw under load:
```bash
nvidia-smi --lock-gpu-clocks=<f>,<f>
# Run a representative workload (e.g., a few vLLM inference batches)
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits
# Collect multiple readings, take average
nvidia-smi --reset-gpu-clocks
```

**Step 3**: Fit a polynomial model. Under standard DVFS models, `P(f) ∝ f^k` with `k ≥ 2`. Try:
```
P(f) = p_2 · f^2 + p_1 · f + p_0     (quadratic)
P(f) = p_3 · f^3 + p_2 · f^2 + p_1 · f + p_0   (cubic, if better fit)
```
Or simply use a lookup table mapping each discrete frequency to measured power.

Record the P(f) model/table in the output files.

---

## Algorithm: Energy-Aware Batch Scheduling (Full Specification)

This section provides the **complete, self-contained** specification of the scheduling algorithm to implement. Read it carefully — this is the only reference.

---

### 1. Background and Motivation

In standard LLM serving (e.g., vLLM's default continuous batching), the scheduler uses a simple FCFS policy and the GPU always runs at maximum frequency. This wastes energy when the workload is light or when SLO deadlines are loose.

The energy-aware scheduler explicitly trades off between:
- **SLO adherence**: meeting per-request TTFT (Time To First Token) and TPOT (Time Per Output Token) targets.
- **Energy consumption**: reducing GPU power by lowering GPU frequency when possible.

The scheduler operates at the **per-iteration (per-batch) level**. Each iteration corresponds to one forward pass of the model. Each output token of a request requires exactly one iteration. The scheduler jointly decides: (a) which requests to include in the batch, and (b) at what GPU frequency to run.

---

### 2. System State at Each Iteration

At iteration i, the system maintains two queues:

- **Running queue R_i**: requests that have **already generated their first output token** (i.e., decode requests). Each decode request processes exactly 1 token per iteration (ℓ_{i,n} = 1).
- **Waiting queue W_i**: requests that have **not yet generated their first output token** (i.e., prefill requests). Each prefill request processes all its input tokens in one iteration (ℓ_{i,n} = input length; we do not consider chunked prefill).

The scheduler selects a subset B ⊆ R_i ∪ W_i to form the batch for this iteration.

---

### 3. Per-Request Definitions

For each request n ∈ R_i ∪ W_i at iteration i:

#### 3.1 Token count ℓ_{i,n}

```
ℓ_{i,n} = 1                      if n ∈ R_i (decode request)
ℓ_{i,n} = input_length(n)        if n ∈ W_i (prefill request)
```

#### 3.2 Per-request processing time t_{n,f}

Using the pre-fitted latency model (see "Pre-fitted Batch Latency Model Parameters" section above):

```
t_{n,f} = (a_p · ℓ²_{i,n} + b_p · ℓ_{i,n} · ℓ^{kv}_{i,n} + c_p · ℓ_{i,n}) / f    if n ∈ W_i (prefill)
t_{n,f} = (a_d · ℓ^{kv}_{i,n} + b_d · ℓ_{i,n}) / f^α                                if n ∈ R_i (decode)
```

where f is the GPU graphics clock frequency in MHz.

#### 3.3 Batch execution time ET_i(B, f)

```
ET_i(B, f) = t_c + Σ_{n ∈ B} t_{n,f}
```

This is the additive model: total time = constant overhead + sum of per-request times.

#### 3.4 Baseline utility r_n

Each request has an urgency-based utility:

```
r_n = w_n · w_TTFT    if the next output token index is 1 (n is in W_i, i.e., prefill)
r_n = w_n · w_TPOT    otherwise (n is in R_i, i.e., decode)
```

where:
- `w_n`: per-request priority weight. **Set w_n = 1 for all requests in our experiments.**
- `w_TTFT`: global balanced weight for the TTFT requirement (tunable parameter).
- `w_TPOT`: global balanced weight for the TPOT requirement (tunable parameter).

#### 3.5 Deadline and slack

Each request n has an SLO target:

```
deadline_n = TTFT_n    if the next output token index is 1 (prefill)
deadline_n = TPOT_n    otherwise (decode)
```

**Note: different requests can have different TTFT_n and TPOT_n values.**

The **waiting time** `T_{i,n}` is the time elapsed since request n was last executed (or since arrival, if it has never been executed). Specifically:
- For a prefill request: T_{i,n} = current_time - arrival_time (it has never been executed).
- For a decode request: T_{i,n} = current_time - time_of_last_token_generation.

The **slack** of request n is:

```
s_n = T_{i,n} - deadline_n
```

Interpretation:
- `s_n < 0`: request still has time before its deadline. E.g., s_n = -100ms means 100ms of slack remaining.
- `s_n ≥ 0`: request has already exceeded its deadline by s_n milliseconds.
- A larger (more positive) s_n means the request is more urgent.

---

### 4. Optimization Problem

The scheduler solves the following optimization at each iteration i:

```
max_{B, f}  { Σ_{n ∈ B} r_n · I{ET_i(B, f) ≤ s_n} - β · P(f) · ET_i(B, f) }

s.t.   Σ_{n ∈ B} ℓ_{i,n} ≤ L_max                                              (token budget constraint)
```

where:
- `I{ET_i(B,f) ≤ s_n}` is the **indicator function**: equals 1 if the batch execution time does not exceed request n's slack (meaning: if we finish this batch within s_n time, then request n meets its per-iteration SLO); equals 0 otherwise.
- `β ≥ 0` is the **energy penalty coefficient** (tunable parameter). Higher β means the scheduler tries harder to save energy at the cost of SLO.
- `P(f)` is the GPU power consumption (Watts) at frequency f.
- `L_max` is the maximum number of tokens processable in one batch (GPU memory constraint). Use vLLM's `max_num_batched_tokens` setting.

**Intuition**: The first term rewards including requests that will meet their SLO. The second term penalizes energy consumption (power × time). The scheduler balances these two objectives via β.

---

### 5. Solution Algorithm: Enumeration over Frequency and Threshold

The key insight is that although B and f are coupled, the problem can be decomposed by **enumerating** over a finite set of GPU frequencies and a finite set of execution-time thresholds.

#### 5.1 Define the threshold set T

```
T = {s_n : n ∈ R_i ∪ W_i}
```

This is the set of all distinct slack values across all candidate requests. We interpret each τ ∈ T as a candidate **execution-time threshold**: we restrict attention to batches whose execution time does not exceed τ.

**Why this works**: The indicator I{ET_i(B,f) ≤ s_n} can only change value when ET_i crosses one of the slack values {s_n}. Between two consecutive values in T, the set of requests that can earn reward (those with s_n ≥ τ) does not change, and tightening τ only reduces the feasible region. So it always suffices to search over τ ∈ T.

More formally, if two thresholds τ_1, τ_2 satisfy [τ_1, τ_2] ∩ {s_n : n ∈ R_i ∪ W_i} = ∅, then the set of reward-eligible requests is the same for both. Since decreasing τ within such an interval only tightens the time-budget constraint Σ t_{n,f} ≤ τ - t_c without making any new request reward-eligible, there always exists a breakpoint τ̂ ∈ T with τ ≤ τ̂ that achieves an objective at least as good.

#### 5.2 Define the frequency set F

```
F = {all supported GPU graphics clock frequencies}
```

Query via `nvidia-smi -q -d SUPPORTED_CLOCKS`. This is a finite set (typically 10-30 values).

#### 5.3 Candidate filtering for a given τ

For a given threshold τ, any request n with s_n < τ can **never** contribute positive reward, because:

```
ET_i(B, f) ≤ τ   ⟹   I{ET_i(B,f) ≤ s_n} = 0   whenever s_n < τ
```

So for a given τ, it suffices to consider only the **candidate set**:

```
N(τ) = {n ∈ R_i ∪ W_i : s_n ≥ τ}
```

#### 5.4 Subproblem: 2D Knapsack for fixed (f, τ)

For each fixed frequency f ∈ F and threshold τ ∈ T, solve:

```
B^{(f,τ)} = argmax_{B ⊆ N(τ)}  Σ_{n ∈ B} (r_n - β · P(f) · t_{n,f})

s.t.   Σ_{n ∈ B} ℓ_{i,n}  ≤  L_max              (token budget)
       Σ_{n ∈ B} t_{n,f}   ≤  τ - t_c             (time budget)
```

**Derivation**: The original objective for fixed (f, τ) with the restriction ET_i(B,f) ≤ τ is:

```
Σ_{n∈B} r_n · I{ET_i(B,f) ≤ s_n} - β · P(f) · ET_i(B,f)
```

Since we restrict to B ⊆ N(τ) and ET_i(B,f) ≤ τ, all requests in B have s_n ≥ τ ≥ ET_i(B,f), so I{ET_i(B,f) ≤ s_n} = 1 for all n ∈ B. Expanding ET_i:

```
= Σ_{n∈B} r_n - β · P(f) · (Σ_{n∈B} t_{n,f} + t_c)
= Σ_{n∈B} (r_n - β · P(f) · t_{n,f}) - β · P(f) · t_c
```

The term `-β · P(f) · t_c` is a constant w.r.t. B, so the subproblem reduces to the knapsack above.

**Structure**: This is a **two-constraint (2D) knapsack problem**:
- Each item n has: value = `r_n - β · P(f) · t_{n,f}`, weight₁ = `ℓ_{i,n}`, weight₂ = `t_{n,f}`
- Capacity₁ = `L_max`, Capacity₂ = `τ - t_c`
- Items with **negative value** should be excluded immediately (they can only hurt the objective).

**Solving the 2D knapsack**: Use one of:
- **Greedy heuristic** (recommended for speed): sort by value density, greedily add. See implementation below.
- **Dynamic programming**: discretize the time dimension and solve exactly. Use if |N(τ)| is small.

#### 5.5 Evaluate the true objective

After solving the subproblem, evaluate the **actual objective** (not the knapsack surrogate):

```
J(f, τ) = Σ_{n ∈ B^{(f,τ)}} r_n · I{ET_i(B^{(f,τ)}, f) ≤ s_n} - β · P(f) · ET_i(B^{(f,τ)}, f)
```

This evaluation uses the real indicator function and real ET_i (including t_c).

#### 5.6 Select the best (f, τ) pair

```
τ*(f) = argmax_{τ ∈ T}   J(f, τ)           for each f ∈ F
f*    = argmax_{f ∈ F}    J(f, τ*(f))
B*    = B^{(f*, τ*(f*))}
```

#### 5.7 Execute

Lock the GPU to frequency f*, execute the forward pass with batch B*, then unlock.

---

### 6. Complete Per-Iteration Algorithm (Pseudocode)

```python
def energy_aware_schedule(R_i, W_i, latency_params, P_f_table, beta, w_TTFT, w_TPOT, L_max, t_c):
    """
    R_i: list of decode requests (running queue)
    W_i: list of prefill requests (waiting queue)
    latency_params: {a_p, b_p, c_p, a_d, b_d, alpha}
    P_f_table: dict mapping frequency f -> power P(f) in Watts
    beta: energy penalty coefficient
    w_TTFT, w_TPOT: SLO balanced weights
    L_max: max tokens per batch
    t_c: constant overhead (ms)

    Returns: (B_star, f_star) — the selected batch and GPU frequency
    """

    all_requests = list(R_i) + list(W_i)
    if not all_requests:
        return [], None

    current_time = time.time()

    # --- Step 1: Compute per-request attributes ---
    for n in all_requests:
        # Determine if prefill or decode
        n.is_prefill = (n in W_i)

        # Token count
        n.ell = n.input_length if n.is_prefill else 1

        # Baseline utility
        n.r_n = (1.0 * w_TTFT) if n.is_prefill else (1.0 * w_TPOT)
        # (w_n = 1 for all requests in our experiments)

        # Deadline
        n.deadline = n.TTFT_target if n.is_prefill else n.TPOT_target

        # Waiting time T_{i,n}: time since last execution (or arrival)
        n.T_in = current_time - n.last_executed_time

        # Slack
        n.s_n = n.T_in - n.deadline

    # --- Step 2: Build threshold set T ---
    T_set = sorted(set(n.s_n for n in all_requests))

    # --- Step 3: Get supported frequencies F ---
    F_set = sorted(P_f_table.keys())

    # --- Step 4: Enumerate (f, τ) and solve subproblems ---
    best_J = float('-inf')
    B_star = []
    f_star = F_set[-1]  # default to max frequency

    for f in F_set:
        P_f = P_f_table[f]

        # Precompute t_{n,f} for all requests at this frequency
        for n in all_requests:
            if n.is_prefill:
                n.t_nf = (latency_params['a_p'] * n.ell**2
                        + latency_params['b_p'] * n.ell * n.kv_cache_len
                        + latency_params['c_p'] * n.ell) / f
            else:
                n.t_nf = (latency_params['a_d'] * n.kv_cache_len
                        + latency_params['b_d'] * n.ell) / f**latency_params['alpha']

        for tau in T_set:
            # --- Step 4a: Filter candidate set N(τ) ---
            candidates = [n for n in all_requests if n.s_n >= tau]
            if not candidates:
                continue

            # --- Step 4b: Solve 2D knapsack ---
            time_budget = tau - t_c
            if time_budget <= 0:
                continue

            B_f_tau = solve_2d_knapsack(candidates, L_max, time_budget, beta, P_f)

            # --- Step 4c: Evaluate true objective J(f, τ) ---
            if not B_f_tau:
                J_val = 0.0
            else:
                ET = t_c + sum(n.t_nf for n in B_f_tau)
                reward = sum(n.r_n for n in B_f_tau if ET <= n.s_n)
                energy_penalty = beta * P_f * ET
                J_val = reward - energy_penalty

            if J_val > best_J:
                best_J = J_val
                B_star = B_f_tau
                f_star = f

    return B_star, f_star


def solve_2d_knapsack(candidates, L_max, time_budget, beta, P_f):
    """
    Greedy heuristic for the 2D knapsack subproblem.

    Item n has:
      - value:   v_n = r_n - beta * P_f * t_{n,f}
      - weight1: ℓ_{i,n} (token count)
      - weight2: t_{n,f}  (processing time)

    Constraints:
      - Σ weight1 ≤ L_max
      - Σ weight2 ≤ time_budget
    """
    # Compute adjusted value
    valued_candidates = []
    for n in candidates:
        v_n = n.r_n - beta * P_f * n.t_nf
        if v_n > 0:  # exclude negative-value items
            valued_candidates.append((n, v_n))

    if not valued_candidates:
        return []

    # Sort by value density (value / combined resource usage)
    valued_candidates.sort(
        key=lambda x: x[1] / (x[0].ell + x[0].t_nf + 1e-9),
        reverse=True
    )

    B = []
    used_tokens = 0
    used_time = 0.0

    for n, v_n in valued_candidates:
        if used_tokens + n.ell <= L_max and used_time + n.t_nf <= time_budget:
            B.append(n)
            used_tokens += n.ell
            used_time += n.t_nf

    return B
```

**Complexity**: For each iteration, the algorithm runs in O(|F| × |T| × |N| log|N|) where |N| = |R_i ∪ W_i|. Since |F| is typically ≤30, |T| ≤ |N|, and |N| is the queue size, this is practical for real-time scheduling.

---

### 7. Practical Implementation Notes

1. **2D Knapsack solver**: The greedy heuristic above is the recommended starting point. If you want higher accuracy, implement DP with discretized time dimension. For DP, discretize the time budget into e.g., 1000 bins.

2. **GPU frequency switching**: Use `nvidia-smi --lock-gpu-clocks=f,f` before each batch execution and `nvidia-smi --reset-gpu-clocks` after. This can be done programmatically via `subprocess` or `pynvml`. Note: frequency switching has latency (~10-50ms); account for this overhead or amortize it. If f_star hasn't changed from the previous iteration, skip the switch.

3. **Integration point in vLLM**: The algorithm replaces vLLM's scheduler's `_schedule()` method (or equivalent). You need to:
   - Intercept after the scheduler identifies the candidate requests (R_i and W_i).
   - Compute s_n, r_n, t_{n,f} for each candidate.
   - Run the optimization to select B* and f*.
   - Feed B* back into vLLM's execution pipeline.
   - Set GPU frequency to f* before the forward pass.

4. **L_max**: This is the maximum number of tokens that can be processed in one batch, constrained by GPU memory. Use vLLM's existing `max_num_batched_tokens` config or compute from model/GPU memory.

5. **Tracking T_{i,n} (waiting time)**: You need to maintain a per-request timer that tracks accumulated waiting time since the last execution.
   - For prefill requests: T_{i,n} = current_time - arrival_time.
   - For decode requests: T_{i,n} = current_time - timestamp_of_last_token_generation.
   - After a request is executed in a batch, update its `last_executed_time` to the current time.

---

## Experiment Design

### Baseline: vLLM Default Scheduler

Run vLLM with its default scheduling algorithm (FCFS-based continuous batching) at the **maximum GPU frequency** (default behavior). Record per-request TTFT, TPOT, and total energy consumption.

### Treatment: Alternative Formulation 2 Scheduler

Run the same workload with the new energy-aware scheduler under various parameter configurations.

### Parameter Configurations to Sweep

All experiments use **w_n = 1** for all requests.

#### Balanced weight pairs (w_TTFT, w_TPOT):

| Config ID | w_TTFT | w_TPOT |
|-----------|--------|--------|
| W1 | 1.0 | 1.0 |
| W2 | 2.0 | 1.0 |
| W3 | 1.0 | 2.0 |
| W4 | 5.0 | 1.0 |
| W5 | 1.0 | 5.0 |
| W6 | 3.0 | 3.0 |

#### Energy penalty β values:

| Config ID | β |
|-----------|---|
| B1 | 0.0 (pure SLO, no energy penalty) |
| B2 | 0.001 |
| B3 | 0.01 |
| B4 | 0.1 |
| B5 | 1.0 |
| B6 | 10.0 |

**Total configurations**: 6 (w pairs) × 6 (β values) = **36 experiments** + 1 baseline = **37 runs**.

### Workload Design

Generate a synthetic workload with **diverse TTFT and TPOT requirements**. Create a set of requests (e.g., 100-200 requests) with varied characteristics:

| Request Group | Count | Input Length | TTFT_n (ms) | TPOT_n (ms) | Max Output Tokens | Description |
|--------------|-------|-------------|-------------|-------------|------------------|-------------|
| A (Latency-sensitive short) | 20-30 | 64-128 | 200-500 | 50-100 | 64-128 | Chatbot-like, tight SLO |
| B (Latency-sensitive long) | 15-25 | 256-512 | 500-1500 | 50-150 | 128-256 | Long-prompt chat, tight SLO |
| C (Moderate short) | 20-30 | 64-256 | 1000-3000 | 100-300 | 64-256 | Moderate SLO |
| D (Moderate long) | 15-25 | 512-1024 | 2000-5000 | 150-400 | 128-512 | Long context, moderate SLO |
| E (Relaxed/batch) | 20-30 | 128-1024 | 5000-15000 | 300-1000 | 256-1024 | Batch processing, relaxed SLO |
| F (Mixed extreme) | 10-15 | 32-2048 | 100-20000 | 30-2000 | 32-1024 | Stress test with extreme diversity |

Requests should arrive following a Poisson process with a moderate arrival rate (e.g., mean inter-arrival time = 200-500ms) to create realistic scheduling pressure. Use a fixed random seed for reproducibility.

### Metrics to Collect

For **each experiment configuration**, record:

1. **Per-request metrics**:
   - Actual TTFT (ms)
   - Actual TPOT (ms) — average across all output tokens
   - TTFT SLO met? (1/0)
   - TPOT SLO met? (1/0)

2. **Aggregate metrics**:
   - **Mean TTFT** (ms) across all requests
   - **Mean TPOT** (ms) across all requests
   - **P50/P90/P99 TTFT** (ms)
   - **P50/P90/P99 TPOT** (ms)
   - **TTFT SLO adherence rate** (% of requests meeting TTFT)
   - **TPOT SLO adherence rate** (% of requests meeting TPOT)
   - **Total energy consumption** (Joules) = Σ over all iterations of P(f_i) × ET_i
   - **Average GPU frequency used** (MHz)
   - **Total wall-clock time** (ms) to complete all requests

3. **Comparison deltas** (relative to baseline):
   - TTFT increase: `(mean_TTFT_new - mean_TTFT_baseline) / mean_TTFT_baseline × 100%`
   - TPOT increase: `(mean_TPOT_new - mean_TPOT_baseline) / mean_TPOT_baseline × 100%`
   - Energy reduction: `(energy_baseline - energy_new) / energy_baseline × 100%`
   - TTFT SLO adherence change (percentage point difference)
   - TPOT SLO adherence change (percentage point difference)

---

## Output Deliverables

### 1. `experiment_results.md` (in current working directory)

Must contain clearly formatted tables. Structure:

```markdown
# Energy-Aware Scheduling Experiment Results

## Hardware & Setup
- GPU model: ...
- Supported frequencies: [...]
- P(f) model: ...
- LLM model: ...
- Total requests: ...
- Arrival pattern: ...

## Fitted Latency Parameters Used
(copy the parameter table from above)

## Baseline Results (vLLM Default Scheduler)

| Metric | Value |
|--------|-------|
| Mean TTFT | ... ms |
| Mean TPOT | ... ms |
| P50/P90/P99 TTFT | .../.../ ... ms |
| P50/P90/P99 TPOT | .../.../ ... ms |
| TTFT SLO adherence | ...% |
| TPOT SLO adherence | ...% |
| Total energy | ... J |
| GPU frequency | ... MHz (fixed at max) |
| Total wall-clock time | ... ms |

## Main Comparison Table

| w_TTFT | w_TPOT | β | Mean TTFT (ms) | TTFT Δ% | Mean TPOT (ms) | TPOT Δ% | TTFT SLO% | TPOT SLO% | Energy (J) | Energy Δ% | Avg Freq (MHz) |
|--------|--------|---|---------------|---------|---------------|---------|-----------|-----------|------------|-----------|---------------|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
(one row per experiment configuration)

## Effect of β (Energy Penalty) — Fixing w_TTFT=1, w_TPOT=1

| β | Mean TTFT (ms) | TTFT Δ% | Mean TPOT (ms) | TPOT Δ% | TTFT SLO% | TPOT SLO% | Energy (J) | Energy Δ% |
|---|---------------|---------|---------------|---------|-----------|-----------|------------|-----------|
| 0.0 | ... | ... | ... | ... | ... | ... | ... | ... |
| 0.001 | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Effect of w_TTFT/w_TPOT — Fixing β=0.01

| w_TTFT | w_TPOT | Mean TTFT (ms) | TTFT Δ% | Mean TPOT (ms) | TPOT Δ% | TTFT SLO% | TPOT SLO% | Energy (J) | Energy Δ% |
|--------|--------|---------------|---------|---------------|---------|-----------|-----------|------------|-----------|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Key Findings
(Summarize the trade-off patterns observed: how β controls energy vs. SLO, how w_TTFT/w_TPOT bias the scheduler toward different SLO types, etc.)
```

### 2. `experiment.md` (in current working directory)

Must contain the **complete reproducible workflow** so that on a new server, given fitted latency parameters, one can directly reproduce the experiments.

Structure:

```markdown
# Energy-Aware Scheduling Experiment — Reproducible Workflow

## Prerequisites
- vLLM installed at <path> (specify version/commit)
- Model path (parameterized)
- Python packages needed
- GPU with DVFS support

## vLLM Modifications

### File 1: <path relative to vLLM root>
<Description of what was changed and why>
<Provide the complete modified file content or a clear diff/patch>

### File 2: ...
(repeat for all modified files)

### New Files Added:
<List all new files added to vLLM and their full content>

## How to Apply Modifications
<Step-by-step instructions or a script that applies all modifications>

## Experiment Scripts

### workload_generator.py
<Complete script that generates the synthetic workload>

### run_experiment.py
<Complete script that:
  1. Takes fitted parameters as input (a_p, b_p, c_p, a_d, b_d, α, t_c)
  2. Takes P(f) model as input
  3. Runs baseline + all 36 configurations
  4. Collects all metrics
  5. Generates experiment_results.md>

### power_profiler.py
<Complete script that measures P(f) for each supported GPU frequency>

## Quick Start
```bash
conda activate myvllm

# Step 1: Profile P(f) (only needed once per GPU)
python power_profiler.py --output power_model.json

# Step 2: Run all experiments
python run_experiment.py \
  --model-path /path/to/model \
  --power-model power_model.json \
  --a_p 6.974e-02 --b_p 0 --c_p 3.395e+02 \
  --a_d 2.112e-01 --b_d 4.822e+01 \
  --alpha 0.817 --t_c 57.05 \
  --output experiment_results.md
```

## Switching Between Default and Energy-Aware Scheduler
<Instructions for toggling between the two schedulers, e.g., via an environment variable or config flag>
```

---

## Implementation Guidance

### Where to Modify in vLLM

The key integration points (explore the codebase to confirm exact locations):

1. **Scheduler** (`vllm/core/scheduler.py` or similar):
   - This is where batch selection happens. The default scheduler uses FCFS with continuous batching.
   - You need to add an alternative scheduling path that implements the Alt. Formulation 2 algorithm.
   - Make it switchable via a config flag (e.g., `--scheduling-policy energy_aware` vs. `default`).

2. **Scheduler output / Sequence group metadata**:
   - You need access to each request's: waiting time T_{i,n}, whether it's prefill or decode, ℓ_{i,n}, ℓ^{kv}_{i,n}, TTFT_n, TPOT_n.
   - TTFT_n and TPOT_n are custom SLO parameters — you'll need to add these as fields to the request metadata. Attach them when requests arrive.

3. **GPU frequency control**:
   - Add a module that can lock/unlock GPU frequency before/after each forward pass.
   - Use `pynvml` or `subprocess` calls to `nvidia-smi`.
   - The frequency should be set BEFORE the batch is executed and reset AFTER.

4. **Energy tracking**:
   - Track cumulative energy: E_total += P(f*) × ET_i(B*, f*) for each iteration.
   - Use the latency model to estimate ET_i, or measure actual wall-clock time.

5. **SLO tracking**:
   - For each request, record actual TTFT (time from arrival to first token output) and actual TPOT (average time between consecutive output tokens).
   - Compare against the request's SLO targets.

### Summary of the Algorithm's Key Properties

1. **Decomposition**: The joint (B, f) optimization is decomposed by enumerating over the finite sets F (frequencies) and T (slack breakpoints), reducing each subproblem to a 2D knapsack.
2. **Threshold justification**: Only τ ∈ T need to be checked because the indicator function structure only changes at breakpoints. Interior points between breakpoints can never improve the objective.
3. **Energy-SLO trade-off**: β controls the trade-off. When β = 0, the scheduler maximizes SLO adherence only. As β increases, the scheduler favors lower frequencies and smaller batches to save energy, at the cost of potentially missing SLO targets.
4. **Starvation avoidance**: Requests that have waited a long time have large (positive) s_n, making them eligible under more τ values and giving them higher priority through the indicator function.

---

## Important Notes

1. **Fair comparison**: Both the baseline and the energy-aware scheduler must process the **exact same set of requests** with the **exact same arrival times**. Use a replay-based approach: pre-generate the workload trace (arrival times + request parameters), then replay it for each configuration.

2. **GPU frequency switching overhead**: Switching frequency takes ~10-50ms. This is non-trivial. You may want to either:
   - Include this overhead in the measurements (more realistic).
   - Amortize by not switching every iteration if the optimal f hasn't changed.
   - Document the approach chosen.

3. **Default scheduler energy measurement**: For the baseline, the GPU runs at max frequency. Energy = P(f_max) × total_wall_clock_time.

4. **Warm-up**: Discard the first few iterations of each experiment run.

5. **Reproducibility**: Use fixed random seeds. Record vLLM commit hash, GPU model, driver version.

6. **Timeout**: If a request has been waiting for too long (e.g., 10× its SLO target), it should still be scheduled eventually. The algorithm naturally handles this because its slack s_n grows, giving it higher urgency through the indicator function — but verify this in practice.

7. **Edge case — empty batch**: If no request has positive adjusted value for any (f, τ) combination, the scheduler should still make progress. Fall back to scheduling the highest-priority request at minimum frequency to avoid starvation.

8. **Restore vLLM**: After experiments, either revert your modifications or ensure they are cleanly toggle-able. Document how to switch between default and energy-aware scheduling.
