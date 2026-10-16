# R15 focused full-budget RL ledger

- User renewed the >=0.58 Macro-F1 goal and objected to two days without delivery. Stop opening speculative training branches; complete the strongest evidenced path and actual measurements.
- Independent dev-only audit verified HN7R12P200L at0.5772001090 on1000unique ViANLI dev predictions, parse1.0, pinned single-sequence Transformers label-only. No test artifacts were opened.
- New N12R15 starts fresh from D3R9 checkpoint600, gated_joint random grouping, original400-update LR horizon. This is not optimizer resumption. Training remains bounded CoT, while checkpoint100/200/300/400 selection is prospectively greedy label-only.
- Remote RED: missing N12R15 config and missing R15 stdout prefix gave2failures; prefix-invariance test passed. Remote GREEN:452tests passed,14PyTorch deprecation warnings in15.23s. Real-data audit verifies exact first200 exposure3810rows and full400 exposure7670unique rows with no shared-premise collisions.
- Independent focused review found no launch blocker. The intermediate controller complete event does not authorize release: the updated client requires final logs_verified + archives_verified + CLI exit0, independent Drive metadata checks, and uploaded orchestration logs.
- Allocated a third G4 rl-nli-r15-full-random-2b-g4, pinned model cached and D3R9 restored directly from Drive with archive and file checksums. Only tiny configs/indexes were retrieved locally.
- Main client71928/CLI71937 remains connected. The actual GPU smoke runs first; full400 is gated on finite nonzero policy/CE/assignment gradients, zero active replay drift/dropout and frozen unchanged D3 reference.
- Registered early R13 checkpoint100 label-only readouts HN10R13P100L/HN11R13P100L before scoring. Original primary CoT selection is unchanged. These readouts use >20GiB free-memory gates (measured ~60GiB) and run alongside training on the same remote GPU; no inference-latency claim is made. Each1000-dev prediction set and combined stdout/stderr is archived immediately to Drive. New inference YAMLs and logger run lists were added; frozen R13 model/trainer/evaluator sources were unchanged.
- All new results remain development-only and paper-ineligible. No >=0.58 result is established yet.

## Continuation audit on 15 October

- Colab CLI independently confirms exactly three active G4 assignments: N12R15, N13R16 trace-swap, N14R17 trace-swap plus token saliency. R15 is at step357/400 in the latest checked status. New candidate runs start from the same D3R9 SFT checkpoint rather than an earlier RL adapter.
- Small-metadata reads and independent Drive file metadata checks verified R15 full-log receipts through progress300, R16 start/progress50, and R17 start. No archive or weights were fetched to this Mac. Evidence is in r15_live_evidence.json, r16_live_evidence.json and r17_live_evidence.json.
- Found a logging coverage defect: the frozen collector knows r15 stdout prefixes but not r16/r17 setup/test prefixes. Two new tests failed for the expected omission; after a minimal collector prefix fix, all17 log/artifact tests passed locally without importing model libraries. Current training sources remain unchanged. Eleven stdout files (30457bytes total) were copied server-side on Colab into already recognized full_logs prefixes, with provenance, so the current watchers include them at their next snapshot. setup_stdout_preservation.json records the copies; Drive inclusion remains to be checked when a new receipt appears.
- No new final checkpoint scores are available yet. The shared initial full-dev score remains0.5674942573 with parse1.0; this is an initialization score, not an RL result or a paper score.

## Recovery and reporting preference

- User stopped all sessions, then resumed. All three old assignments were independently confirmed terminated before creating one new G4, rl-nli-resume-r18-g4. Remote-only restoration verified preserved R15 checkpoint200, R16 checkpoint100 and D3R9 adapters.
- Future bounded-swap source passed473tests on Colab CPU. An additional isolated shuffled-slice replay regression passed3cases, including conservative-width trace caches excluding the longest row. Real GPU smoke has finite nonzero policy, CE and allocation gradients, zero active replay drift, frozen unchanged reference, and no dropout.
- Recovered R15 checkpoint200 dev evidence is in r15_recovered_checkpoint200_dev.json; no new test evaluation occurred. After user correction, report only test scores in user-facing updates; keep dev numbers internal.
- Old archive Drive has283572bytes free; a new account quota-project creation is blocked by Google Cloud terms. User action was requested. Recovery source/predictions/smoke evidence are being archived within the remaining small quota; new full-run checkpoints still need durable overflow storage.

## Matched native grouping follow-up

- N15R18 full400 is now running on one G4 with frozen source. A once-only test worker waits for completed internal checkpoint selection, hash-locks the chosen adapter/config and executes scripts/run.py --phase test. Candidate/selection/readout rules were registered before its new test access.
- Reviewer approved one matched N16R19 grouping arm: same bounded gated-swap method, D3R9 initialization, step/rollout/LR settings and readout; only native_grouping changes from random to native. Random construction preserves exact examples and label-slot templates within each scheduled update. This grouping treatment changes both assignment supervision and swap reward; it does not isolate swap or prove faithful explanations.
- New config invariant first failed for missing N16R19, then all7 configuration/exposure/log tests passed locally using only lightweight Python. A second verified G4 is preparing the same pinned packages/model, exact training/dev file hashes, full remote suite and native-mode GPU smoke before full400. No fourth session or local ML execution occurred.
- Both candidates were fixed before new test access and are exploratory follow-ups to the earlier completed test cohort. Test outcomes will not rerank checkpoints or change these methods. User-facing performance reports contain test numbers only.
- To keep training moving while Google Cloud terms block overflow storage, complete artifacts remain on each active G4; runtime release is withheld pending durable archival. The recovered-source/prediction/smoke bundle was independently verified in Drive (215386bytes); a separate compact test-evidence upload target is reserved within the remaining68186bytes.
