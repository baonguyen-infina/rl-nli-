# Full-budget random-group RL implementation plan

> Execute inline using superpowers:executing-plans. The user already authorizes autonomous teacher-free SFT+RL research and G4 experiments toward greedy dev Macro-F1 >=0.58.

**Goal:** Prospectively test the complete400-update horizon of the best credible earlier RL label-only development view.

**Architecture:** Start a fresh run N12R15 from the exact D3R9 SFT checkpoint600. Reuse gated_joint RL with random, gold-multiset-matched training groups; regenerate all400 updates and verify the first200 updates preserve the earlier exposure. Training remains bounded CoT; evaluation and checkpoint selection use the frozen single-sequence Transformers label-only interface. No new algorithm or exact optimizer continuation is claimed.

**Tech stack:** Qwen3.5-2B pinned15852e8c16360a2fea060d615a32b45270f8a8fc; BF16 LoRA; TRL1.15.0; Transformers5.17.0; vLLM0.31.0; G4 only.

**Spec:** Continuing user request, AGENTS.md, and the design in this document. R13 remains frozen on its two existing VMs.

## Constraints and design

- No teacher, distillation, synthetic rationales, UID features, local ML, new test access or per-cell hyperparameter tuning.
- Reference remains the D3R9 SFT adapter, tensor fingerprint662232c84b36759aba1e7260ff2febf9c544f73f5d62ab9343afd7d55359677f.
- Existing full_reasoning shared profile:400 updates, LR2e-6, K4, eight groups/update, reasoning128/total160, score microbatch12, KL.01, gold CE.1, direct assignment loss.2, save100. Only remove the earlier stop200 by selecting that existing profile.
- Output inference is greedy label-only; checkpoints100/200/300/400 are evaluated on all1000dev rows, best Macro-F1 wins, ties later. Warm label-only dev is recorded first. This new protocol is explicitly posthoc development research; earlier N7R12 is not relabeled as label-selected.
- Assignment conditions on observed gold label multisets, including repeats/singletons. Gold enters supervised loss and random-control construction, never the prompt. It does not force one of each label, establish a new theorem or show native semantic grouping helps.
- Drive headroom was1,173,982,007bytes before this run. Preserve a compact checkpoint200 adapter plus final selected adapter and complete logs; other temporary checkpoint weights remain only on Colab through evaluation. Existing two R13 cohorts already have reserved backup targets; avoid extra duplicate weight archives.

## Review focus

- Adapter initialization must not silently become an optimizer resume or reset reference to a previous RL checkpoint.
- Full random schedule must preserve the first200 updates and expose no duplicate training row.
- Evaluation format must be label-only while the native trainer still explicitly generates bounded CoT.
- All actual runs go through scripts.run with frozen config; full-dev count/metrics/source hashes must be verified.
- All available stdout/stderr/rollouts and failure logs stay remote and get checksummed archives before GPU release.

## Tasks

- [x] Add remote-first configuration/exposure tests and observe missing-run RED.
- [x] Register N12R15 and one-update G0N12R15P in configs/matrix.py using existing base profiles; no trainer/optimizer change.
- [x] Run full remote regression and real-data first200-prefix exposure audit.
- [x] Run one-update GPU smoke requiring finite nonzero policy, CE and assignment gradients, frozen reference, zero dropout and zero active replay drift.
- [x] Start fresh400-update main run only after all gates pass.
- [ ] Keep client/kernel connected through completion, save full logs remotely and monitor actual observations.
- [ ] Verify final full-dev results from1000unique predictions and archive receipts; do not write dev results into paper or open test.
