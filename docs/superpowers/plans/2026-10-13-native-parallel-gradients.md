# Native reward and direct gradient parallel pilot implementation plan

> **For agentic workers:** Use superpowers:executing-plans for integration and remote execution. The isolated mathematical helper is delegated to allocation_math; it may edit only its helper and tests.

**Goal:** Compare native allocation reward, direct allocation gradients, and their combination with teacher-free SFT + RL on seed0 full dev, aiming at .57–.58 Macro-F1.

**Architecture:** Every branch starts from the identical selected D3R3 adapter. The native objective conditions label assignment on the observed gold multiset for siblings sharing premise text. Detached allocation gradients are replayed through canonical answer scores per microbatch, avoiding a full generation-batch model graph. GRPO remains an approximate clipped, normalized surrogate; the combined branch is not claimed to be an unbiased joint gradient estimator.

**Tech Stack:** Qwen3.5-2B, TRL1.15.0, vLLM0.31.0, Transformers5.17.0, PEFT0.21.1, three independent Colab G4 runtimes.

**Spec:** User authorization to research teacher-free SFT + RL methods and run three G4s concurrently, together with AGENTS.md data and evaluation safeguards and 2026-10-13-native-rl-pilot.md.

## Global constraints

- No teacher, teacher rationale, distillation, or synthesized gold premise/hypothesis.
- No test split; greedy dev1000 chooses checkpoints; seed0 only at this stage.
- No uid, intended-label, topic, or premise-ID features. Grouping uses normalized premise text.
- All runs use scripts.run; hyperparameters in base.yaml; method differences in matrix.py.
- No local training, model loading, heavy tests, or model downloads. Restore Drive directly to Colab.
- Fixed LR2e-6, 200 steps, eight groups/update, four rollouts/item, 64 reasoning tokens, 96 total tokens, KL.01, gold answer CE.1, allocation coefficient.2, disabled RL dropout.
- Immutable run IDs/configs, source snapshots, gradients/reference/dropout audits and verified Drive archives.

## Review focus

- TRL shuffles and splits the generation batch: cached allocation Jacobians must travel with their exact prefix rows.
- Padding and same-label cohorts: zero allocation gradients; include zero-loss groups in the fixed group normalizer.
- Score replay: identical full canonical label suffix, EOS, no length normalization, no active dropout; audit detached/live numerical differences.
- Auxiliary normalization: summing all microbatch contributions must reproduce a group-mean objective without a second gradient-accumulation division.
- GPU lifecycle: finish and verify archives before stopping each session; do not mutate N2R4 while it trains.

## Prespecified variants

| Run | Allocation reward | Direct allocation loss | Other objectives |
|---|---|---|---|
| N2R4 | .2 logQ | none | accuracy+.2 format GRPO, .1 answer CE, KL.01 |
| N4R4 | none | .2 mean(-logQ) | same |
| N5R4 | .2 logQ | .2 mean(-logQ) | same |

The direct channel supplies the derivative of label scores holding sampled reasoning fixed; the reward channel changes the sampled policy via GRPO advantages. Their roles are distinct. Final dev performance and matched controls determine whether either helps; an allocation objective alone is not a novelty claim.

## Task 1: Launch corrected reward-only pilot

- [x] Allocate rl-nli-native-clean-g4 with the newly authenticated compute account.
- [x] Upload source-only archive and private Drive restore manifest.
- [x] Launch pinned dependency setup, train/dev preparation, full190 tests, verified D3R3 adapter restoration.
- [x] Prepare fourteen private upload targets and queue N2R4 via scripts.run.
- [ ] Confirm setup and first policy/CE gradients, initial dev, full200 updates, selected dev and immutable reference.

## Task 2: Allocation linearization helper

Create train/native_allocation.py and tests/test_native_allocation.py.

Interfaces: allocation_linearization(scores, members, gold, weights) returns detached scores, Jacobian, per-row value share and scalar group_count. allocation_surrogate(live, old, jac, share, count) returns `(jac*(live-old)).sum()/count + share.sum()/count`.

- [x] Upload tests first; observe24 remote failures on missing helper, then24 passes.
- [x] Implement helper using tensor_log_allocation on detached score leaves with torch.enable_grad.
- [x] Verify gradients and values against the direct group-mean objective, arbitrary shuffled slices, padding, empty/same-label cohorts and fixed-multiset bias invariance.

## Task 3: GRPO integration

Modify train/native_grpo.py, configs/base.yaml, configs/matrix.py; create generated N4R4/N5R4 and G0N5/G0N6 YAML configs. Create tests/test_native_allocation_trainer.py with a tiny CPU model and an isolated GRPO test double, running only on Colab.

- [x] Observe5 method failures on remote CPU (missing tensor cache and reward-channel separation), then6 passes.
- [x] Compute detached scores for native/direct/joint modes; use allocation reward only for native/random/joint.
- [x] Cache linearization tensors and explicit answer-prefix tensors into the generated batch; recover prefixes using their attention mask, including padding rows.
- [x] Score the three canonical labels with gradients in training mode and disabled dropout, using score microbatch4 and checkpointing.
- [x] Add globally group-normalized direct loss and first-step gradient/replay audit. Preserve old reward-only branches.
- [ ] Full remote regression suite, then one-update GPU smoke for each new mode with nonzero policy/CE/allocation gradients and unchanged reference.

## Task 4: Execute three lanes

- [x] Allocate rl-nli-native-direct-g4 and rl-nli-native-joint-g4; confirmed3 assignments at26.70 CU/hour.
- [x] Upload identical final source, pinned setup and direct Drive warm start; both pass220 tests. Queue G0N5/N4R4 and G0N6/N5R4 respectively.
- [ ] Start per-lane archive watcher; poll sparsely; select checkpoint100/200 by greedy full-dev Macro-F1.
- [ ] Fetch verified metadata only; compare initial reasoning, selected reasoning and label-SFT warm start with distinct interfaces stated.
- [ ] Stop each GPU after verified complete and metadata archives. Decide subsequent matched controls from measured method results, without opening test data or claiming acceptance/SOTA.

## Completed legacy-scheduler pilot outcomes

| Run | Initial reasoning dev Macro-F1 | Checkpoint100 | Checkpoint200 | Selected |
|---|---|---|---|---|
| N2R4 | .51732278 | .53992841 | .54895447 | 200 |
| N4R4 | .53434450 | .55802922 | .55757327 | 100 |
| N5R4 | .52813231 | .55652164 | .54718175 | 100 |

All selected parse rates are1.0. No pilot reaches .57; these are one-seed dev results under the original scheduler. Reference preservation and archives are being finalized, and fixed-runtime validation remains in progress. Do not infer a meaningful improvement over D3R3 label-SFT (.55283311) from a few thousandths of Macro-F1 with the observed inference variation.

Shared inference ablation added: V2R4L, V4R4L and V5R4L evaluate each learned adapter using the original label-only prompt under the same fixed runtime as VDR4L. They run after each lane's reasoning validation, and complete/metadata archives are required before GPU termination. This checks whether RL training improves direct classification; it adds no training targets and changes no pilot weights.

Runtime correction after Drive inspection: The first clean runtime disappeared after starting N2R4, and its durable logs reached step60; the earlier interpretation that it failed before training was wrong. Initial and progress20/40/60 archives were verified, recovered as metadata only, and preserved under N2R4_attempt1 names on Drive. No checkpoint was available for resume. The replacement starts fresh from the same D3R3 adapter and records execution_provenance.json as attempt2; its upload capabilities were replaced to avoid reusing completed attempt1 sessions. The failed assistant helper agent delivered tests before exhausting its workspace credits; root completed helper/integration after observing RED on Colab. No local model/test execution was used.

Current validation concern: nominally greedy warm-start reasoning dev Macro-F1 differs across runtimes (.51732, .53434, .52813), while the LoRA warm-start gradient audits agree closely. vLLM documents batch invariance separately, and upstream issue48613 discusses limitations for GDN attention. This suggests a backend reproducibility issue but does not prove its cause in our runs. Preserve all current pilot code/selection; validate repeated fixed-adapter evaluation before drawing small-gain conclusions. The replay difference metric currently includes padded rows whose cached scores are intentionally zero, so its overall maximum is not an active-score discrepancy measurement; check active coordinates separately in subsequent validation.

Normalization audit: Installed TRL _BaseTrainer sets compute_loss_func to a non-None sentinel for trainers whose loss_is_scaled_for_ga is true on Transformers<5.19. DAPO uses the full generation-window token normalizer; shuffle/split preserve scalar normalizers and permute row tensors together. No additional gradient-accumulation division is introduced by the parent trainer under this compatibility path.

## Fixed-adapter validation queued after each pilot

- Added a shared validation_eval preset: max_num_seqs1, max_num_batched_tokens2048, eager execution, prefix caching off, seed0, greedy decoding. Do not assert bitwise determinism before checking it.
- Added optional eval engine controls after8 RED failures, then229 total GREEN tests on all three validation checkouts.
- Validation runs V2R4A/B, V4R4A/B, V5R4A/B repeat each selected method adapter twice. VDR4A/B repeat the unchanged warm adapter with the reasoning interface; VDR4L checks its label interface under the same runtime preset.
- These are isolated source checkouts, with existing remote results/data paths shared. Current pilot training and legacy checkpoint selection code remain immutable.
- All validation runs use scripts.run; preserve each result, not the maximum over repeats. Each GPU stops only after primary and validation complete/metadata archives are verified.

## Related-work boundaries for interpreting this pilot

- GTA (Findings EMNLP2025): https://aclanthology.org/2025.findings-emnlp.56/ combines supervised initial guesses with RL on reasoning/final answers and a gradient conflict rule. Merely combining CE and GRPO is therefore not a novelty claim for this project.
- Instance bundles (EMNLP2021): https://aclanthology.org/2021.emnlp-main.584/ already uses related examples and contrastive normalization. Our conditional assignment mathematics must be positioned relative to this precedent.
- SoftmaxGRPO (August2026 preprint): https://arxiv.org/abs/2608.09271 studies alternatives to standardized group advantages. Its finite-group objective caveat reinforces avoiding an unbiased-objective claim for our clipped GRPO broadcast reward.
- RL-ZVP (ICLR2026 author page): https://bltnynk.github.io/publications/rl-zvp/ addresses prompts with identical rollout rewards. Dense native supervision alone is insufficient novelty; native/random grouping and RL/no-RL controls are needed if the pilot succeeds.
- Batch invariance documentation: https://docs.vllm.ai/en/latest/features/batch_invariance/ and GDN feature discussion: https://github.com/vllm-project/vllm/issues/48613. Do not enable an unsupported determinism flag without validation.

Replay diagnostic at the unchanged D3R3 warm checkpoint: four actual student rollout texts were reencoded identically for both forward modes, with one row deliberately treated as inactive. Active canonical-score max difference was0.0; the unmasked max was3.85253 entirely from zeroing the inactive cached score. This supports the padding explanation for the logged discrepancy metric; it is a four-prefix diagnostic, not a proof covering every training batch. No optimizer step was performed. Result is attached to N2R4/replay_diagnostic.json for archival.

Further related-work check: https://arxiv.org/abs/2602.03979 (Likelihood-Based Reward Designs, February2026) already studies answer log-probability rewards, while https://arxiv.org/abs/2405.19763 (RLLR, ACL2024) already studies label-sensitive RL for NLU. Neither a likelihood reward nor label-sensitive pairs alone should be claimed as our novel contribution.
