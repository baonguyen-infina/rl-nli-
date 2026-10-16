# Native token credit pilot implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement and verify each task on Colab. No heavy local execution.

**Goal:** Test whether native assignment sensitivity can improve RL credit allocation beyond the .5580 seed0 development pilot, aiming at .57–.58 Macro-F1.

**Architecture:** Retain gold-label SFT initialization, accuracy/format GRPO, answer CE and direct native assignment loss. Compute attribution to each sampled reasoning token from the detached native assignment score Jacobian times gradients of canonical answer scores with respect to that token's input embedding. Sum the three label contributions before taking absolute magnitude, normalize token credits within each reasoning span, and keep generated answer credit at1. This is a biased token-weighted GRPO surrogate; attribution measures local sensitivity and does not prove rationale faithfulness.

**Tech Stack:** Qwen3.5-2B, TRL1.15.0, BF16 LoRA, three Colab G4s, source and verified archives only on remote machines.

**Spec:** User authorization to autonomously research teacher-free SFT + RL methods and run3 G4s concurrently; AGENTS.md data safety and run entrypoint rules; 2026-10-13-native-parallel-gradients.md.

## Global constraints

- No teacher model, generated rationale targets, pseudo labels, UID features, test data, or local model downloads.
- Keep the same D3R3 adapter, seed0, data order,200 updates, eight whole groups/update, four rollouts/item, LR2e-6,64-token reasoning budget, KL.01, answer CE.1 and allocation loss.2.
- N1R5 is ordinary correctness/format RL+CE; N4R5 repeats direct allocation+RL; N7R5 adds native token credit to N4R5. These are matched controls for one method hypothesis.
- Saliency floor.25 is shared in base.yaml. Never choose per-cell hyperparameters.
- Exclude padding and inserted syntax. Preserve each reasoning span's total credit; use uniform credit when contrast sensitivity is zero.
- Keep pilot checkpoint selection and separately report fixed-runtime reasoning and label-only inference checks. Record all experiments and no maximum over inference repeats.

## Review focus

- Cancel label contributions at the embedding-vector level before taking magnitude.
- Do not accumulate model parameter gradients during credit estimation.
- Token positions must use exact sampled IDs and left-padding offsets; delimiters and prompt text receive no reasoning credit.
- Tensor advantages must survive TRL shuffle/split; rollout logging must retain the original scalar sequence advantage.
- Gradient sensitivity is not causal explanation quality, and the technique must be cited relative to GRAIL (https://arxiv.org/abs/2606.04889).

## Task 1: Credit mathematics

Create train/native_saliency.py and tests/test_native_saliency.py.

- [x] Write and observe remote RED for label-vector summation, cancellation, padding, varying reasoning lengths, zero-signal fallback, normalized mean1 and finite nonnegative credits.
- [x] Implement normalized credit using `w=floor+(1-floor)*L*s/s.sum()` on each active reasoning span, with uniform fallback. All other positions are zero in the stored reasoning-credit tensor.
- [x] Run remote GREEN and compare against a hand-computed example. No local tests.

## Task 2: Trainer transport and GPU scoring

Modify train/native_grpo.py and tests/test_native_allocation_trainer.py.

- [x] Observe remote RED for saliency mode cache/transport and per-token advantages.
- [x] Reuse allocation score Jacobians; replay canonical label scores with `inputs_embeds` requiring gradients; call autograd.grad only on embeddings and collect the sampled-reasoning slices.
- [x] Sum gradient-activation contributions across the three candidate labels, normalize with the shared floor, and expand scalar GRPO advantages with reasoning weights.
- [x] Keep scalar advantages in logs, record credit summaries and source provenance. Preserve existing modes.
- [x] Full remote regression suite and a one-update GPU smoke proving finite credits, nonzero policy/CE/allocation gradients and unchanged reference.

## Task 3: Matched200-step pilot

Modify configs/base.yaml/configs/matrix.py; generate N1R5/N4R5/N7R5 and a saliency smoke config.

- [x] Freeze all3 configs before launch. Upload source without models to available remote runtimes.
- [ ] Run each via scripts.run, archive checkpoints100/200 and selected adapter, compare full-dev Macro-F1 and both inference formats.
- [ ] If there is no improvement, report it and use the measured failure to choose the next authorized method iteration. No publication/SOTA/acceptance claim follows from a single development seed.

## Verified launch evidence

On 2026-10-14 Vietnam time, all three isolated R5 checkouts passed248 remote tests. G0N7 completed its GPU update with policy gradient norm0.18328, answer CE0.15980 and allocation0.41411. Reasoning credit mean1.0, standard deviation0.55340, range0.35089–7.78463; no parameter gradients accumulated during credit estimation. The frozen reference SHA remained unchanged. N1R5, N4R5 and N7R5 were queued on separate G4 runtimes, with prior validation and label inference completing before the next GPU job. Their200-step results are still pending.
