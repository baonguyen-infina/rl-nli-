# Label-marginal RL implementation plan

> Use superpowers:executing-plans inline. The user has authorized autonomous teacher-free SFT+RL research and remote experiments; no additional design approval is required.

**Goal:** Test a substantially different credit signal for the continuing seed0, 2B ViANLI experiment, targeting greedy dev Macro-F1 at least0.58.

**Architecture:** Reuse the gold-only D3R9 SFT checkpoint and the existing bounded reasoning policy. Score all three canonical answer suffixes after each sampled reasoning prefix. Marginalize the answer choice in the reward and in a direct score-gradient replay. The primary new pilot uses the leave-one-example change in minibatch Macro-F1 as the label utility; a matched accuracy-utility variant distinguishes metric coupling from answer marginalization. Sample other examples' canonical labels from the detached three-class distribution with a recorded seed. Mask sampled answer tokens from the policy loss so their gradient is not counted twice. Train only sampled reasoning tokens with leave-one-out advantages; retain the existing gold answer CE and frozen SFT KL reference.

**Tech stack:** Qwen/Qwen3.5-2B, BF16 LoRA, TRL1.15.0, Transformers5.17.0, vLLM0.31.0, Colab G4. Shared400-update LR horizon with a200-update pilot stop, K4, eight premise blocks/update, LR2e-6, KL0.01, answer CE0.1, direct utility loss0.2, reasoning128/total160. Keep native blocks only to preserve prior exposure: the new reward imposes no native label assignment constraint.

**Spec:** User's continuing SFT+RL research authorization, AGENTS.md, and the design below. No teacher/distillation, model above3B, local ML execution, UID features, or new test access.

## Design and alternatives

1. More native-allocation reward/attribution variants are deferred: the completed study did not establish their benefit.
2. Individual answer likelihood is a useful established comparator, not sufficient novelty: likelihood-based reasoning rewards and their direct supervised gradient are already studied in https://arxiv.org/html/2602.03979v1 .
3. Selected pilot: exact finite-label conditional utility with a training-batch Macro-F1 reward. This is a hypothesis about dense credit and metric alignment, not an established novel algorithm or guaranteed estimator of greedy corpus Macro-F1.

For example i and reasoning sample k, define q_i,k=softmax(s_i,k) over the three canonical suffix sequence log probabilities. For each k, independently sample canonical answers a_j,k from detached q_j,k. Remove example i from that rollout's confusion matrix C_-i,k. Set U_i,k(c)=B*(MF1(C_-i,k+e_gold_i,c)-MF1(C_-i,k)), where B is the active distinct-example count. The utility depends on other sampled examples but not on i's sampled answer or reasoning. Conditional reward r_i,k=sum_c q_i,k(c)U_i,k(c); direct loss is -r_i,k. Cache its exact derivative with respect to the three scores and replay with the existing fixed-width mechanism. Accuracy utility instead uses U(c)=1[c=gold].

Use A_i,k=r_i,k-mean_{l!=k}r_i,l, without group standard-deviation scaling. Forced syntax, padding and sampled answers receive no policy/KL tokens. This is a clipped, token-normalized policy surrogate, not an unbiased implementation of a sequence-summed expected corpus F1 gradient. q is a constrained canonical distribution, whereas reported inference remains the existing greedy generative HF protocol. These distinctions must remain explicit.

## Review focus

- A candidate's utility must not depend on that candidate's sampled answer; changing its gold changes only the analytically verifiable confusion contribution.
- Padding is excluded before softmax/metrics; inactive NaN scores cannot poison rewards or gradients.
- All K trajectories of an example have the same gold; grouping uses existing label-blind metadata, never fields in the prompt.
- Cached/live replay must preserve the exact global scoring width across shuffled microbatches; compare analytic gradients with brute-force expected utilities on synthetic cases.
- RLOO groups by example, not the three sibling hypotheses. Mask answers before TRL calculates token KL/importance ratios.

## Tasks

- [x] Add remote-first tests in tests/test_label_marginal.py: counterfactual F1 versus exhaustive confusion reconstruction; own-answer independence; sample reproducibility; padding/NaNs; cached/direct gradients and arbitrary shuffled slices; correct RLOO groups.
- [x] Implement train/label_marginal.py with utilities, detached linearization and RLOO. Validate shapes, labels, binary masks and finite active scores.
- [x] Extend the actual reward/rollout/loss hooks in train/native_grpo.py and train/native_rollout.py for two new modes, with integration tests in tests/test_native_allocation_trainer.py. Preserve old modes unchanged.
- [x] Register N10R13 (Macro-F1 utility), N11R13 (accuracy utility), and one-update G0 smokes. All runs through scripts/run.py, all hyperparameters in base.yaml, all selection by full-dev greedy bounded-CoT Macro-F1 using Transformers. Label-only is a prespecified secondary view.
- [x] Run remote RED/GREEN and the existing remote suite. Require finite nonzero policy/direct/CE gradients, unchanged frozen reference, zero active dropout and zero active cache/replay score drift before full training.
- [ ] Back up selected/terminal adapters and metadata to the existing private Drive, without downloading weights locally. Record source/config/adapter hashes and compute consumption. Compare actual dev results before deciding further research.

## Reporting boundary

The official test was opened for the earlier frozen cohort. Do not access or analyze it to tune these pilots. New dev/training measurements remain internal and do not enter paper tables, figures or empirical discussion. A later confirmatory claim needs a separately justified evaluation protocol; passing0.58 on dev alone does not establish0.58 on test, novelty, or A/A* readiness.

Verification: remote helper9-test RED/GREEN, actual trainer2-test RED, rollout-mask RED, then437 remote tests passed in8.27s on primary and10.22s on the matched variant after its dependency readiness race was corrected. Independent read-only review found no critical/important issue; conditional-gradient validity is not a corpus-F1 improvement claim. Both real GPU one-update smokes passed all gradient/reference/dropout/replay gates. Both main pipelines are running initial full-dev inference before the200-update pilot; no new main-run score yet.
