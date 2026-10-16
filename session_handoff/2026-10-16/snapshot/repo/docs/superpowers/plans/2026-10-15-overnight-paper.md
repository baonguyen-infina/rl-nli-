# Overnight native-reward and test-only paper implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans for inline execution. User has explicitly authorized autonomous research/implementation/Colab loops while asleep; do not add another approval round.

**Goal:** Complete a finite teacher-free 2B SFT/RL study and an honest ACL working paper by the morning of15October, with experimental results in the manuscript exclusively from a locked held-out test evaluation.

**Architecture:** Preserve frozen R9/R10 code and checkpoints. Compare native allocation, ordinary RL and joint reward; the next method replaces unbounded logQ policy shaping with a bounded correctness-gated exp(logQ) bonus. Add a separate greedy Transformers evaluation path to investigate the observed vLLM/GDN reproducibility problem, without rewriting earlier results.

**Tech Stack:** existing Qwen3.5-2B,LoRA,BF16,TRL1.15.0,Transformers5.17.0,vLLM0.31.0,three Colab G4s; local work limited to source, small metadata and document artifacts.

**Spec:** prior authorized SFT/RL research scope; docs/superpowers/plans/2026-10-14-native-r9-config-audit.md and latest user instructions.

## Global constraints

- Model2B instruction/post-trained by default, absolute maximum3B; no4B model operations.
- No teacher queries or distilled rationales. Frozen reference is the same seed's SFT checkpoint already required for KL.
- No local model weights, training, model inference or ML tests; all such work runs in Colab.
- No UID/intended-label information in prompts, scoring inputs or model features.
- Keep test closed while researching. Select candidates and checkpoints with dev; do not target a threshold by looping on test.
- All experiment invocations through scripts/run.py with frozen per-run configs and artifact paths. GPU diagnostics can be separate and must never be reported as benchmark runs.
- User explicitly deferred extra training seeds; any morning results are identified as single-seed evidence, not a multi-seed study or established A/A* result.
- Paper contains only test performance. Dev, training and numerical diagnostics stay in artifacts/analysis and research notes.

## Review focus

1. Hypothesis-only baseline must not see the premise, sampled rationale, gold label or identifier; baseline cache keys derive from prompt token IDs.
2. Reference-adapter context must restore active adapter and requires_grad flags, preserve model weights, and never accumulate reference gradients.
3. Cached/live corrected score Jacobians must match under fixed padding and zero dropout; inactive rows have zero contribution.
4. A different evaluation backend is a separate prespecified protocol; never overwrite or select among inference repeats.
5. Held-out evaluation must validate a manifest with frozen source/config/adapter hashes before reading test. Repeated test invocation must reuse completed results or fail, never tune.

## Tasks

- [ ] Review current runs, primary recent literature, fixed-runtime variance and method claims; preserve a candid publishability assessment.
- [ ] Implement bounded correctness-gated native reward as a bounded extension. Add unit tests for reward bounds, missing/inactive groups, finite/shape validation and explicit no-policy control; run RED/GREEN remotely, then a one-update GPU gate.
- [ ] Run gated joint N6R11 from D3R9 on the first fully released G4 after its existing validation pipeline. Keep R10 running. Use the existing full_reasoning profile; no run-specific LR/reward sweeps.
- [ ] On another released G4, run the decisive no-policy-gradient native allocation control with matched data and budget, subject to implementation/test gates; ordinary RL already exists. Random grouping is secondary if the gated method shows a clear dev improvement and time permits.
- [ ] Investigate deterministic greedy evaluation on32 fixed dev examples using Transformers, with repeated fresh-process outputs. If consistent, register and apply one shared backend/format to all final comparisons before test locking. Do not claim unsupported vLLM batch-invariance flags fix GDN.
- [ ] Rewrite paper/main.tex around actual native assignment method, not the legacy alignment/GTA draft. Include exact mathematical invariance and limitations, related work boundaries and actualBF16/LoRA settings. No invented empirical claims.
- [ ] By05:30Asia/Ho_Chi_Minh stop adding candidates and freeze the method/checkpoint/inference rule using completed dev evidence. If unfinished, exclude those candidates from the locked test cohort rather than reading test early.
- [ ] Implement and verify the G4 test manifest gate and once-only evaluation record. Test matched SFT,ordinary RL,main method and completed decisive ablations under the same locked inference protocol. Keep test sealed for any subsequent research cycle after this report.
- [ ] Generate paper tables from recorded test artifacts, add appropriate paired uncertainty/per-class analysis and limitations, compile ACL PDF remotely or with a lightweight local TeX tool, render and inspect every page.
- [ ] Final evidence audit: only test scores in paper; configs/source/adapter hashes recorded; no unrun ablation, invented seed SD, SOTA or A/A* acceptance claim. Deliver source, PDF and experiment-status report, explicitly naming any incomplete evidence.

## Initial evidence and novelty boundaries

At17:16UTC14October, all R9 mains complete. Native with D3R9 selects step100 with devMF1.5715148029; ordinaryRL selects400 with.5679703759; old-warmnative selects200 with.5655205227. Native-vsordinary gap0.3544 percentage points is below observed greedy repeat variability. JointN5R10 is running after a passed gradient/reference/replay gate. These are internal development results and must not enter paper empirical tables.

Primary boundaries: NLI-GRPO already exists(arxiv2504.18376v3,2026revision); likelihood-based CoT rewards(arxiv2602.03979); contrastive and shortcut mitigation methods(arxiv2603.17305,2604.12277). Calibration by a frozen hypothesis-only baseline is a hypothesis to test, not an established debiasing guarantee or assumed new invention. Conditional assignment cancels row/class offsets but does not remove all hypothesis-specific shortcuts.

The vLLM GDN batch-invariance limitation is documented upstream(issue48613,openedJuly2026); our local installed version and fresh-process tests still need verification. Do not patch external kernels overnight without a minimal validated cause.

Design review decision: inverse-PoE s-b can reward an incorrect full model when the frozen H-only reference is even more wrong. TRAIN-only PoE s+b is more principled but requires a reliable H-only baseline audit and can saturate assignment on biased groups. Defer both. The selected next method is a bounded correctness-gated native Q policy bonus; direct −logQ remains unchanged. With actual base reward a+.2f and bonus≤.2, every correct completion has raw reward≥1 and every wrong completion≤.2. This raw-reward ordering is not a guarantee about group-normalized advantages. No-RL allocation_only keeps dense CE/allocation gradients but zeroes policy/KL loss.
Remote checks: all391 tests passed8.17s, including newHF backend, manifest/receipt guards, bounded reward and actual trainer control integration. REDs observed for new helpers and wiring before implementation. No localMLtests or test data accessed.
