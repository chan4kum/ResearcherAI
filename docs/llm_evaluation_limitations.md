# Known Limitations of LLM Evaluation in Continuous Integration

This document outlines the fundamental operational, statistical, and architectural limitations of evaluating Large Language Models (LLMs) and Agentic AI systems within automated CI/CD pipelines.

---

## 1. Non-Determinism and Sampling Variance

### The Problem
Even when temperature is configured to `0.0`, LLM inference engines in production environments exhibit non-deterministic behavior due to:
- GPU floating-point non-associativity across parallel CUDA warp operations
- Dynamic server-side request batching and speculative decoding
- Periodic model updates and quantization variations on managed provider APIs

### CI Impact & Mitigation
- **Risk**: Flaky CI builds caused by benign stochastic variation in generated phrasing.
- **Platform Mitigation**:
  - We do not enforce brittle string equality; instead, we use semantic keyword sets (`must_contain_any`), forbidden phrase assertions (`must_not_contain`), and bounded word counts.
  - Non-perfect passing thresholds ($\ge 85\%$) and a regression delta tolerance buffer ($5\%$) prevent minor stochastic drift from blocking deployments.

---

## 2. LLM-as-a-Judge Systematic Biases

When using an LLM to evaluate another LLM's outputs, several well-documented cognitive and statistical biases emerge:

| Bias Type | Mechanism | Operational Consequence |
|:---|:---|:---|
| **Verbosity Bias** | LLM judges disproportionately award higher quality scores to lengthy, verbose responses. | Short, precise, direct answers are falsely penalized in favor of bloated essays. |
| **Self-Enhancement Bias** | Models systematically prefer answers synthesized by their own architecture family (e.g. GPT judging GPT vs Claude). | Distorts multi-model benchmark comparisons in model upgrade pipelines. |
| **Position / Ordering Bias** | In pairwise comparative evaluations ($A$ vs $B$), judges favor the first or last presented option. | Introduces systematic scoring skew unless positions are randomized across twin runs. |
| **Egocentric Sycophancy** | Judges rate polite, deferential, or agreement-heavy answers higher than firm, fact-based refutations. | Degrades adversarial safety and boundary-testing evaluations. |

---

## 3. Evaluation Dataset Drift & Benchmark Contamination

### Dataset Staleness
Static test sets degrade over time as application capabilities, enterprise data schemas, and user query patterns evolve. A test case written for v1.0 may become irrelevant or misleading as agents gain new tools and capabilities.

### Contamination
When open benchmarks or production question sets leak into foundation model pretraining or instruction-tuning datasets, benchmark scores appear near-perfect while real-world capability remains unchanged (*memorization vs. generalization*).

---

## 4. Latency and Cost Tradeoffs in Continuous Integration

Running a 500-question evaluation dataset with multi-turn LLM agent execution on every micro-commit in GitHub Actions introduces severe bottlenecks:
- **Cost**: A full run with production frontier models (e.g., GPT-4o) can cost \$5–\$50 per CI build.
- **Latency**: Multi-turn agent execution with retrieval and tool loops can take 15–45 minutes per run.

### The Hybrid Evaluation Pyramid
Our platform resolves this tradeoff using a 3-tier evaluation strategy:

```
          / \
         /   \     Tier 3: Nightly Full Frontier Judge
        /     \    (Golden dataset, multi-turn human baseline)
       /───────\
      /         \   Tier 2: PR Quality Gate (Milestone 48-49)
     /           \  (18 representative cases, mock/fast provider, <5s)
    /─────────────\
   /               \ Tier 1: Unit & Heuristic Tests (Pre-commit)
  /                 \ (315+ deterministic unit tests, pytest, <2s)
```

---

## 5. Groundedness vs. Helpful Reasoning Dilemma

Evaluating RAG groundedness involves a fundamental tension:
- **Strict Groundedness**: Demands that *every single word and entity* in the answer be explicitly present in retrieved context chunks. This causes false failures on trivial deductive steps (e.g., inferring that "Berlin is in Europe" when context states "Berlin is in Germany").
- **Lenient Groundedness**: Allows external world knowledge into the synthesis. This allows subtle hallucinations to slip past automated checkers.

Our evaluator balances this by combining **hallucination phrase detection**, **context token overlap ratios**, and **error disqualifiers**.

---

## 6. Threshold Strategy Summary

| Quality Gate | Standard Threshold | Justification |
|:---|:---:|:---|
| **Overall Score Minimum** | `85.00%` | Ensures high baseline quality while tolerating edge-case variations. |
| **Max Allowed Drop vs Baseline** | `5.00%` | Detects true quality regressions without tripping on stochastic noise. |
| **Max Dimension Drop** | `10.00%` | Flags isolated component degradation (e.g. tool routing failure). |
| **Agent Success Minimum** | `85.00%` | Enforces state-machine and lifecycle integrity across agent runs. |
| **Retrieval Relevance Minimum** | `80.00%` | Guarantees vector search and HyDE maintain citation standards. |
| **Citation Correctness Minimum** | `85.00%` | Asserts schema completeness and provenance traceability. |
