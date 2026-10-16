# Current teacher-free RL cohort: internal review

This note contains development/training evidence only. It is not a paper results source. The current research cycle has not opened test.

## Prospective interpretation

- N12R15: full400-update gated assignment RL, with random groups and greedy single-sequence Transformers label-only checkpoint selection.
- N13R16: a swap-based reward candidate. Relative to R15 it changes the gate, transformation and reward range as well as subtracting swapped-prefix utility; it is not an isolated swap ablation.
- N14R17: swap plus original-allocation token saliency. Compare against R16 checkpoint200 to match exposure and LR horizon. Comparing each run's best endpoint additionally changes selection and budget.
- Saliency differentiates the original allocation objective; it does not attribute the swap reward. Conditional assignment probability also cancels shared class-score offsets, so an improved assignment score alone does not establish better absolute classification calibration.
- All three optimize answer CE and direct assignment gradients in addition to RL. A later policy-off matched control is necessary before crediting gains specifically to RL.

## Verified implementation observations

Both independent source reviewers identified variable padding widths in swapped and saliency scoring, whereas original cache/replay use a fixed global width. This is a possible BF16 numerical confound; no effect magnitude has been measured and it is not a demonstrated semantic masking bug. A future scoring diagnostic must measure it before changing live experiments.

A direct read of installed TRL loss source on both the R15 and R17 VMs found conditional unsqueezing only when advantages.dim()==1 (lines2981–2985), so the suspected unconditional-unsqueeze broadcast problem is absent in both installed copies.

CPU-only audit of already recorded training rollouts, steps0–19, on each VM:

| Run | Mixed-correctness prompt groups | Incorrect samples | Incorrect samples with positive scalar advantage | Correct samples with negative scalar advantage |
| --- | ---: | ---: | ---: | ---: |
| N12R15 | 213 | 379 | 0 | 0 |
| N13R16 | 227 | 374 | 39 | 109 |
| N14R17 | 221 | 374 | 36 | 94 |

Each audit covered386 prompt groups with four rollouts per prompt. Source hashes of the active trainer were unchanged. These observations diagnose reward transport; they neither prove worse held-out performance nor establish rationale faithfulness. Complete diagnostic JSON is written under each remote run directory for the next archive snapshot.

## Logging repair verification

The frozen collectors omitted r16/r17 setup/test stdout prefixes. The local regression reproduced two failures, then17 log/artifact tests passed after the prefix fix. On an isolated source-only Colab copy, the first full-suite attempt failed collection because the temporary copy omitted the teacher source module needed by a unit test. After copying that Python source,455 tests passed with14 existing torch deprecation warnings in8.21seconds. No teacher model, dataset or weights were loaded; active training sources remained unchanged. Both attempt logs are preserved under full_logs_r16traceswap prefixes.

Eleven existing stdout files were copied server-side into prefixes recognized by the running frozen watchers. Their next Drive snapshot must be checked for inclusion; no logger source in a live training checkout was replaced.

## Conditional next experiment

First finish the registered runs and inspect actual checkpoint scores. If needed, test the diagnosis that unrestricted swap contrast weakens correctness-based credit. One bounded alternative is Rbase +0.2*correct*(Qown-Qswap), retaining existing CE/direct-allocation settings and matched exposure. The adjustment is in[-0.2,0.2]; this preserves raw correctness reward dominance, not a guarantee about total gradients or causal explanation quality. No new GPU job is authorized to exceed three concurrent sessions.

Novelty boundaries were checked against primary sources: GTA (2025.findings-emnlp.56), GRPO for NLI (arXiv2504.18376), GRAIL (arXiv2606.04889), In-Context RLVR (2026.findings-acl.1523), and ResRL (PMLR306/lin26ak). Teacher-free SFT+RL, generic likelihood rewards and token saliency are not new by themselves; an eventual numerical threshold does not establish an A/A* contribution.
