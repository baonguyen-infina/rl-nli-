# Native hypothesis contrast RL pilot

> Execution authorized by the user's “oke train đi”. Implement in the current experiment branch; run every test and all model work remotely on Colab G4. No further design approval is needed for this approved pilot.

**Goal:** Train and evaluate seed-zero SFT→RL with native same-premise assignment utility, aiming at 0.57–0.58 greedy development Macro-F1.

**Architecture:** One Qwen3.5-2B BF16 backbone with trainable LoRA, initialized from a reproduced D3 SFT checkpoint. Student-generated short reasoning is rewarded for final correctness, format, and correct assignment of native hypothesis labels. Assignment scores are detached during reward computation; answer-only gold CE is an explicitly separate auxiliary term.

**Tech stack:** PyTorch, Transformers 5.17.0, TRL 1.15.0, vLLM 0.31.0, PEFT 0.21.1; confirm actual remote versions before training.

**Spec:** Approved configuration in the immediately preceding conversation.

## Constraints

- No teacher, distillation, teacher rationales, or invented semantic process labels.
- No test access; GPU G4 is unrelated to the protocol's test gate.
- Prompt inputs contain premise/hypothesis only; UID, topic, intended label and gold remain external metadata.
- All runs enter through `scripts/run.py --run ID --seed 0`; immutable run configs and real results.
- Local machine: source edits and CLI only; no model downloads, tests, plotting, compilation or checkpoint backups.
- RL: LR 2e-6, 200 updates, four rollouts/item, eight groups/update, at most three items/group; token microbatching preserves full groups.
- Reasoning at most 64 tokens and total completion at most 96; greedy dev at steps 0/100/200.
- LoRA rank32/alpha64/dropout0.05; KL0.01 explicitly references D3, not the adapter-disabled base model.
- Final correctness +0.2 format; assignment utility coefficient0.2; gold answer CE coefficient0.1.
- Native/random/ordinary RL share all other settings and original example exposure.

## Review focus

1. Same-label/singleton groups have zero allocation signal and retain ordinary learning.
2. Random groups preserve every active example, native size/label templates and disjoint premise membership.
3. Canonical answer probabilities score entire answer sequences without length normalization or unmasked prompt loss.
4. Gold-label vectors and sibling grouping never become inference inputs.
5. Microbatch boundaries cannot break the native group; padding contributes zero RL, KL and CE.
6. Reference logits use the exact warm-start adapter; no accidental reference to the base model.

## Tasks

- [x] Provision one authorized G4; install the previously working pins; upload source only. First runtime vanished; replacement `rl-nli-native-rl-2` is active.
- [x] Reproduce D3 under D3R and retain its selected adapter remotely: checkpoint300, greedy dev Macro-F1=0.5534757662, accuracy=0.557, parse=1.0. Checkpoint400=0.5500318882.
- [x] Write allocation/grouping tests, demonstrate failure remotely, then implement grouping and log-space assignment likelihood.
- [x] Write helper tests for label suffix scoring, padding masks and group rewards; demonstrate failure remotely before implementation. GPU reference/gradient audit remains pending.
- [x] Implement a grouped generation batch using TRL and an explicit preserved group sampler; detached assignment scores and weighted answer-only CE.
- [x] Add the approved pilot configuration once in base.yaml and named run differences in matrix.py; retain old frozen runs unchanged.
- [x] Run the remote full CPU/Torch test suite and one GPU update.177 tests passed. G0N failed due nested custom-rollout likelihoods; shape regression reproduced and fixed. G0N2 passed one full update: policy gradient norm0.22594, answer-CE norm0.14836, active loss tokens3595, CE tokens400; exact reference-adapter SHA unchanged.
- [ ] Evaluate warm-start label and prospective reasoning prompts; record format transfer separately.
- [ ] Run native method and matched controls, selecting checkpoints only by full dev Macro-F1; write logs, predictions, metrics, costs and provenance.
- [ ] Transfer only small logs/metrics/predictions; monitor remotely and stop the runtime after completed results are secured remotely.

## Mathematical contract

For group G with active members i, canonical class scores s_i(c), and observed gold multiset y:

`logQ = sum_i s_i(y_i) - logsumexp_{a in unique_permutations(y)} sum_i s_i(a_i)`.

Each rollout index defines one independent joint draw across the group. Add coefficient0.2 times logQ to member rewards. Groupwise reward normalization is a declared GRPO surrogate, not an unbiased gradient claim for a joint CE objective. Ordinary RL sets the coefficient to zero. Random allocation uses precomputed matched different-premise blocks. Gold answer CE trains only the answer suffix on student prefixes; it never trains against a teacher rationale.

Implementation details established by review: reasoning and answer are sampled in two stages; injected delimiters and EOS have policy mask zero. Retained vLLM stop IDs are removed before continuing. Random grouping is matched inside each scheduled eight-group update cohort, preserving the exact exposure at steps100/200. TRL receives the entire generation batch24 times, allowing its buffer to return24 aligned microbatches. Runtime asserts this schedule. The complete remote test suite passed176 tests, including regression tests for both stop IDs and sampler buffering. Versions verified: Torch2.13.0, TRL1.15.0, Transformers5.17.0, PEFT0.21.1, vLLM0.31.0, flash-linear-attention0.5.2.

Rollout inspection: read50 active outputs from G0N2 (64 active/96 padded completions). Initial sampling parse=0.90625 and64-token reasoning cap fraction=0.59375. Outputs include mixed languages, hallucinated details and partial reasoning; format validity alone is not semantic faithfulness. Inspection exposed retained stop markers sharing a token with trailing punctuation and redundant sampled opening tags. N2 was interrupted after10 updates before any selected post-RL metric; its artifacts are archived as aborted. Two failing regression fixtures reproduce these boundaries. A corrected shared decoder censors at decoded marker starts and excludes redundant structural tags from the two stages. Fresh G0N3 smoke and N2R pilot retain exactly the same optimization hyperparameters and D3R initialization. Future matched controls use this same decoder; do not compare archived broken-decoder N2 against corrected controls.

## Stop and advancement rules

Abort numerical failures, wrong-reference/sync bugs or persistent invalid formats. A reasoning prompt drop is reported as an interface tax, not credited to later RL. Finish the prespecified 200-step pilot before expanding; only advance a method that improves independent greedy dev over SFT and matched controls with parse≥0.99. The target and budget gates are not publication acceptance guarantees.

## Runtime recovery and durable artifacts, 2026-10-13

Both earlier Colab assignments disappeared during the interruption. The final N2R metric and adapter were not retrieved; do not infer a final score from its first training batch or initial evaluation. A new G4 assignment, `rl-nli-durable-g4`, runs fresh IDs D3R3 and N2R3 with the unchanged approved optimization configuration. N2R3 initializes from D3R3; any future matched controls must use that same warm start and corrected bounded decoder.

The current remote suite passed 186 tests, including seven artifact-upload regressions. D3R3 is training. Its checkpoint100 archive has been uploaded directly from Colab to the private Drive experiment folder, acknowledged with size and MD5 verification (120822290 bytes; file ID `1QqS9KZFHWtNTUH7VQ-ilEl7V6bJe9B-S`). No model files were transferred to the local machine.

D3R3 subsequently completed in 844.9 seconds. Greedy full-dev checkpoint Macro-F1 values were 0.5143822813 (100), 0.4830977859 (200), 0.5484207312 (300), and **0.5528331075 (400, selected)**; final accuracy=0.553 and parse=1.0. This fresh warm start differs from the previous D3R checkpoint300; all matched follow-up runs must therefore initialize from D3R3. Its final adapter is archived remotely. Only the 83180-byte metadata archive was retrieved locally with an MD5 check; the receipt and real metrics are under `results/D3R3_Qwen3.5-2B_s0/`. N2R3 has started its pre-RL reasoning evaluation.

N2R3 pre-RL greedy reasoning evaluation: Macro-F1=0.5241706756, accuracy=0.526, parse=1.0. First-update gradient audit: policy norm=0.2156371514, answer-CE norm=0.1631792702. Reference initialization SHA=`ef2a456431eb6fc665d619f69a8ccaff99e095fa2b747f06d74c26191719df2c`. The full RL checkpoint100 archive was verified on Drive (467059413 bytes, MD5 `da34672d6f8cd620e41708118e630561`). Final post-RL dev score remains pending.

### Dropout audit and fresh scoring protocol

Installed TRL1.15.0 defaults `disable_dropout=False`. Transformers calls `model.train()` before GRPO input generation/scoring; the detached score path did not switch to evaluation mode. D3R3's LoRA dropout=0.05 therefore adds noise to assignment scores and policy/reference likelihoods. This was missed in the initial audit. N2R3 continues its full prespecified200 updates without changing its loaded code or frozen config and must be reported as the stochastic-scoring pilot. Before any edits, its six training/eval source files were copied into `source_snapshot.json` and checked against startup hashes; `implementation_audit.json` records the finding.

Future native runs explicitly disable dropout in GRPO and score detached rewards in evaluation mode, restoring every module's prior mode afterward. Four CPU regressions were observed failing before the helper existed, then passing; the full remote suite passed190 tests. Fresh IDs N0R4/N1R4/N2R4/N3R4 all initialize D3R3 and share the same corrected scoring protocol and other native hyperparameters. Missing dropout flags in old frozen configs retain the original False behavior. G0N4 is a single-update GPU verification scheduled after N2R3 and its final archives finish; it is not an extra performance baseline. Its metadata is also archived to Drive. The lightweight local stop controller now waits for both main-pilot and G0N4 archives before releasing the G4. GPU verification and new full deterministic pilots remain pending.

### Completed R3 pilot and deterministic GPU verification

N2R3 completed in3195.0 seconds. Full-dev greedy Macro-F1 was0.5435721531 at checkpoint100 and **0.5520696302 at checkpoint200 (selected)**, with accuracy0.552 and parse1.0. This is+2.7899 Macro-F1 percentage points over its pre-RL reasoning prompt, but0.0763 points below D3R3 label SFT; it does not establish a method gain over SFT or reach the0.57–0.58 target. The reference SHA stayed unchanged. Final archives are durable on Drive; only the598342-byte metadata archive was retrieved locally.

G0N4 completed its single GPU update in66.9 seconds. Actual dropout audit: requested=True, active dropout modules={}, detached eval reward scoring. Policy gradient norm=0.2659290930, answer-CE norm=0.1596978035; the exact D3R3 reference SHA stayed unchanged. Its58959-byte metadata archive was retrieved with an MD5 check. The local controller then terminated `rl-nli-durable-g4` after all four main/smoke final archives were confirmed durable. No active GPU assignment remains. The account has3.23 compute units, about22 minutes of G4, insufficient for another full200-update pilot plus setup. N2R4 is the next main run; matched controls and multi-seed work have not been launched.

Checkpoint restore for the next VM must go directly from Drive to Colab. Drive's LRO download URI rejected an anonymous range request (HTTP403); a short-lived Drive-file-only OAuth token succeeded on an authenticated one-byte range probe (HTTP206), without any local model file. Private credential/restore manifests remain in `/private/tmp`, outside repository/configs/artifacts; never print or commit them. A separate private Drive credential state preserves artifact access if the Colab compute account changes. The next setup script should refresh the scoped restore token, upload the private manifest separately, verify archive size/MD5 on the VM, restore D3R3 checkpoint400, and run N2R4 with the corrected shared scoring configuration. Restore implementation is prepared but full remote restore has not run yet.

`/content/durable_pipeline.py` runs D3R3, waits for its verified final archive, then runs N2R3 through `scripts.run`. A remote artifact watcher persists checkpoints, progress metadata, final selected adapters and small metadata-only archives. The file-scoped upload manifest stays outside source/config/results. A lightweight local HTTP/CLI controller (`/private/tmp/native_autostop.py`) stops only this G4 after both N2R3 final archives have nonzero sizes and Drive MD5s; it never downloads weights. Monitor `/content/durable_pipeline.log`, `/content/archive_watch.log`, `/content/d3r3.log`, and `/content/n2r3.log`. The account balance was 13.63 compute units at the last check, with an 8.90/hour rate. New D3R3/N2R3 dev scores remain pending.
