# Native rationale-swap pilot

## Decision and motivation

The user authorizes autonomous teacher-free SFT+RL research on three Colab G4s. R5 remains running. Its token-credit lane has53.33% zero-reward-variance groups over the most recent20 updates at step45. Token weights cannot recover a zero scalar advantage. Prepare a dense group reward that tests whether sampled reasoning helps distinguish the original human hypotheses in its own context, relative to reasoning rotated among those hypotheses. Launch depends on the R5 results and completed prior GPU jobs; do not overlap evaluators and trainers on one GPU.

## Method

Keep the same D3R3 gold-label SFT adapter,200 updates, seed0, native premise-text blocks, K4, LR2e-6, KL.01, answer CE.1, direct assignment loss.2, and64-token reasoning budget. For each complete native block and rollout index, rotate student reasoning prefixes by one active slot, leaving each premise-hypothesis prompt unchanged. Rotation uses slot membership only, excluding zero-weight padding; it never consults labels, UID, intended labels or topics. Singletons retain their own prefix and give zero contrast signal.

Let Q be the existing exact posterior over unique assignments of the observed training-label multiset. Score Q with the original prefixes and with the rotated prefixes using the current student, detached and with dropout disabled. Use `U=log Q(original)-log Q(rotated)` and the existing shared allocation reward coefficient.2. Retain the original direct assignment-gradient loss and gold-answer CE. No rationale is turned into an SFT target. No synthetic premise, hypothesis or label is generated. At inference, the model sees one premise-hypothesis pair and generates greedily.

This reward measures context sensitivity of sampled reasoning; it does not establish semantic validity, causal faithfulness or unbiased policy-gradient estimation. Incorrect verbalized answers inside reasoning can drive the contrast. Inspect late rollouts and compare with logQ reward alone before making a publication claim.

## Comparisons

- N5R6: repeat direct assignment+logQ reward, controlling for a dense joint group reward.
- N8R6: replace that logQ reward with original-minus-rotated logQ.
- N9R6: N8R6 plus the already tested native token-credit mechanism, if the R5 findings justify this combination.
- N4R5/N7R5 provide the corresponding correctness-reward controls. Keep every completed and failed run, and separate repeated inference from training seeds.

## Work checklist

- [x] Observe remote RED for label-blind cyclic mapping, padding exclusion, singleton behavior and input validation.
- [x] Implement rotation helper and trainer scoring branch; retain existing-mode behavior.
- [x] Observe remote RED/GREEN for actual reward transport and unchanged direct allocation Jacobians.
- [ ] Run the full remote regression suite and a one-update GPU smoke before launching a new200-step pilot.
- [ ] Freeze matrix configs and source snapshots, prepare verified Drive archive targets, and queue only after each prior validation finishes.
- [ ] Compare full dev Macro-F1, inference repeats and label-only inference; do not choose an inference-repeat maximum.

Remote verification:13 rotation tests and13 real trainer-hook tests passed. The isolated R6 checkout passed268 total tests before GPU smoke; no training weights were loaded locally. N9R6 retains attribution from the original native assignment loss, rather than claiming attribution of the original-minus-rotated reward. R5 training and its frozen source remain untouched.

## Closest recent sources

GRAIL uses gradient-activation weights for outcome RL; R5 adapts that idea to the native assignment objective, so token saliency alone is not a novelty claim: https://arxiv.org/abs/2606.04889 . In-Context RLVR evaluates reasoning through demonstration utility, showing that usefulness-based reasoning supervision already has precedents: https://aclanthology.org/2026.findings-acl.1523/ . GC2PO uses latent episode perturbations and token rewards: https://arxiv.org/abs/2602.06475 . CounterRoute uses paired reasoning/direct rollouts for routing credit: https://arxiv.org/abs/2609.29109 . The proposed experiment instead uses existing human NLI hypotheses sharing a premise and an exact assignment posterior. That distinction requires empirical evidence and further prior-work review, rather than an A/A* guarantee.
