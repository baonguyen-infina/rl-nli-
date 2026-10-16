# R20 internal evidence and resource decisions

This is an internal training/development ledger, never a source for paper results. The 0.58 test Macro-F1 target is not achieved.

## Current decision

Keep the two already registered teacher-free 2B candidates; allocate no third G4 and launch no additional training before reviewing their checkpoint100 development evidence. Frozen 400-step training and final dev checkpoint selection remain unchanged. Checkpoint100 is an evidence/review event, not an implemented automatic early-stop rule.

N20R20 uses full-horizon trace saliency; N21R20 uses full-horizon label-marginal macro-F1 utility. They are alternative mechanisms sharing the SFT warm start, data schedule, and update horizon; their comparison is not a one-factor causal ablation, and compute costs differ.

## Revised diagnosis

The earlier completed runs have finite gradients and moderate entropy changes. Their logs do not support severe policy collapse or near-total zero-variance saturation as the established cause. The initial claim that accuracy versus macro-F1 mismatch was the key cause was too strong: on this balanced benchmark, their numerical difference is very small.

Paired dev-only predictions, selected checkpoints versus the same SFT warm start, give the following descriptive counts. These are checkpoint-selection data, so paired p-values are not confirmatory significance tests.

- N15R18: 38 repairs and 28 regressions. Neutral contributes 6 repairs and 16 regressions; 395 examples remain wrong under both models.
- N16R19: 42 repairs and 30 regressions. Neutral contributes 3 repairs and 21 regressions; 391 examples remain wrong under both models.

These observations support investigating the neutral-versus-contradiction trade-off and unchanged difficult errors, not simply increasing training reward. They do not establish a new method or predict held-out improvement. No test predictions were read for this diagnostic.

## Cheap checkpoint observation

An independent observer waits for a completely saved checkpoint100, verifies all frozen source hashes, checks more than20GiB free GPU memory, and evaluates the same full-dev greedy-label protocol into ckpt_eval/checkpoint-100. Final checkpoint selection reuses these results. It never changes config, training code, cost status, checkpoint scores, or test claims. Both observers were confirmed as live remote processes after launch.

## Preservation

All model loading/inference/archive packaging occurs on Colab. A capped-memory relay saves periodic logs and compact checkpoint adapters into the owned private unpublished GitHub archive release. Final verified test archives remain a separate once-only operator. Operator sources and their SHA256s are retained beside this ledger. No checkpoint or model file is persisted on the Mac.

## First N21 checkpoint review

Checkpoint100 dev predictions are verified (1000 unique rows, unchanged greedy-label protocol). Their paired comparison with the warm start is stored in r20_n21_checkpoint100_internal.json. The aggregate gain is small and neutral regressions remain; this is not compelling evidence that macro-F1 utility solves the observed trade-off.

Before observing checkpoint200, registered a resource-review rule in artifacts/protocols/20261015/N21R20_checkpoint200_review.json: if the best of checkpoints100/200 is below the warm-start full-dev Macro-F1 +0.005, abort this exploratory arm without a test run and preserve its partial artifacts. If it continues, retain original selection over100/200/300/400. This is a compute-allocation heuristic, not statistical evidence. The remote checkpoint200 observer is live (PID81175); the root controller must apply the stopping decision after verified results. There is no automatic termination daemon.

The checkpoint100 adapter is durably archived (GitHub private draft asset644678338,123264984bytes; SHA256b00b55760f6e9624e6bfecf1b735b197861f585015c3b22c6dedaff79bcdb53d). No model bytes were persisted locally. Concurrent early dev readout slows training, so these wall times must not be used for a controlled training-efficiency comparison.

A suspected periodic rollout omission was investigated before changing the packer. Both actual archives contain the main run's rollouts through collect_logs, even though snapshot_files filters that file. No omission fix was needed; the first broad suffix assertion also matched the smoke-run rollout and was corrected to identify the main run exactly.
