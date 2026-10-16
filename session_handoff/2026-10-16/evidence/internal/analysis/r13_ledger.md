# R13 finite-label RL ledger

User goal remains seed0 greedy dev Macro-F1 >=0.58 with a 2B instruction/post-trained LLM, teacher-free SFT+RL, G4 only. No local ML or new test access.

- Google network access restored after execution permissions changed. New Colab account authenticated; verified balance1483.87 CU, zero assignments before this cohort.
- Created rl-nli-dense-2b-g4. Actual hardware: NVIDIA RTX PRO6000 Blackwell Server Edition,97887MiB. All model files and tests are remote.
- Restored D3R9 checkpoint600 directly from Drive; archive121105483bytes, MD575167cef623282ab19ce7a1090cfd313. No local weights.
- Helper RED:9 failures for the missing new module. Helper GREEN:9 passed remotely.
- Actual trainer hook RED:2 failures for missing label-marginal mode integration.
- Actual rollout mask RED: sampled answer token retained; expected [0,1,1,0,0,0], observed [0,1,1,0,1,0]. Fixed before full regression.
- Full remote regression GREEN:437 passed in8.27s. Evidence on VM:/content/native_r13_full_tests.json.
- Registered N10R13 (F1 utility), N11R13 (accuracy utility), one-update smokes, and separate label-only secondary views. Shared200-update pilot with400-update LR horizon; greedy Transformers bounded-CoT full-dev checkpoint selection. Hyperparameters inherited from common base profiles, no per-run tuning.
- Both one-update smoke gates passed on actual G4 GPUs. N10R13: policy gradient norm0.0254944, answer CE norm0.247866, direct utility norm0.309533. N11R13: policy0.00807553, CE0.247852, direct utility0.296796. Both have exactly zero active cached/live score drift, zero active dropout, and frozen/unchanged reference SHA662232c84b36759aba1e7260ff2febf9c544f73f5d62ab9343afd7d55359677f. Large differences on inactive padded rows are masked and have zero Jacobians.
- Both main pipelines are now live: N10R13 parent18008/evaluator18009 and N11R13 parent21053/evaluator21054. They are still evaluating the frozen warm checkpoint on full dev before RL; no new main-run dev score or training update is available yet. Quiet logs reflect single-sequence Transformers generation with final-only metric writes, not evidence of a stalled process.
- The accuracy VM initially attempted its regression gate before dependency installation was ready, yielding21 dependency/import failures. The gate prevented any experiment launch. Preserved the failed log and test report, waited for the explicit dependency-ready flag, reran the full suite (437passed in10.22s), and only then restarted the gated worker. This was an operational readiness race, not an intentional TDD failure or a model result.
- Backups preauthorized as file-scoped Drive capabilities, kept outside the repo and logs. Primary worker will refuse full training unless policy/CE/direct gradients are finite/nonzero, reference is frozen/unchanged, dropout is zero and active cache/replay score drift is exactly zero.
- Ruling: the first experiment stops at200 updates on the existing400-update LR schedule to bound a new method pilot; this is shared across both new utility variants. Existing native-allocation objectives are disabled in both.
- Ruling: do not modify or merge the earlier paper from new dev results. The earlier test cohort is already opened and cannot serve as a fresh confirmatory target for adaptive method development.
- Read-only independent implementation/math review completed by /root/r13_review: no critical or important defect found; reviewed actual cached TRL1.15 mask/normalization transport. No local tests or model execution.

- Durable lightweight local monitor PID53177 polls every60s and checks usage every300s. It writes artifacts/analysis/r13_status.json plus lane observations, verifies actual remote processes, and releases each VM only after terminal pipeline status, verified archive receipts, and no remaining experiment processes. It does not launch an unreviewed next method. Initial new account balance1483.87CU; latest observed1474.87CU with two assignments at17.80CU/hr.

Remaining: observe main RL after initial full-dev inference, inspect checkpoint100/200 greedy full-dev outcomes and secondary views, verify final Drive archives, and fetch only compact metadata. No >=0.58 claim is supported yet.

## Interrupted execution and recovery attempt2

- Server-side `colab sessions` and `usage` now confirm zero old assignments; the previous observer had stopped receiving valid remote data before actual VM loss. Do not describe stale process snapshots as live training. Exact backend reclaim cause is not established.
- Drive metadata proves more progress than the last CLI observation: N10R13 archived step60 and N11R13 step20. Those archives and original smoke metadata were checksum-verified and recovered locally as small metadata only. Both full-dev initial evaluations completed; no trained checkpoint100 was uploaded, and no main-method dev result exists.
- Preserved old observer JSONs and compact evidence under artifacts/analysis/r13_attempt1*. New attempts use separate Drive names containing attempt2, preserving original archives. Repeating the same seed after infrastructure loss is not an additional independent run.
- Reallocated G4 sessions rl-nli-r13-recovery-primary-g4 and rl-nli-r13-recovery-accuracy-g4. All13 registered method/protocol source hashes still match. No model, reward, optimizer, checkpoint selection or decoding setting changed.
- Actual workload launch now remains inside a continuously executing Colab cell and connected CLI client, rather than returning immediately after spawning remote background processes. The cell streams real setup and experiment observations. This is a reliability adjustment, not a verified diagnosis or guarantee against backend reclamation.
- Lightweight local relays77226/77517 and stream monitor77862 write fresh attempt2 state. They execute no local ML. A relay releases its GPU only after explicit remote completion and verified archive receipts; an observation/client failure does not authorize duplicate training or unconditional release.
- Both recovery VMs verified actual RTX PRO6000 Blackwell hardware and passed437 full regression tests (15.06s/14.58s). Both new GPU smokes passed: primary policy0.0255010/CE0.247764/direct0.309675; accuracy policy0.00807570/CE0.247855/direct0.297093; both active replay drift0, reference unchanged, active dropout0. Small hardware-level gradient differences across infrastructure attempts are recorded, not treated as independent seed evidence.
- Both main attempt2 pipelines have now started their initial full-dev inference (parents12323/38698); RL updates follow that evaluation. No method dev result is available yet.
- Metadata finalizer88631 will retrieve only exact, checksum-verified complete-metadata archives using returned Drive file IDs; it rejects model/checkpoint files and test artifacts, preserves earlier local attempt files, recomputes development metrics from1000 unique predictions, checks the100/200 checkpoint-selection record and reference/dropout audits, and writes a development-only verified result. It does not write new empirical paper claims or open test.

## Full log preservation (user request, 2026-10-15)

- Complete available stdout/stderr, dependency/test logs, warnings/errors, training CSV, unfiltered rollout JSONL, configurations, source snapshots, metrics and predictions are archived directly from Colab to private Drive. Model/checkpoint binaries are excluded from these log bundles; adapter backups use the existing separate remote artifact policy.
- Archiving events: start, updates50/100/150/200, terminal. Every payload has a SHA256 and byte count in logs_manifest.json. Upload completion verifies the archive MD5 and byte count. The two start archives were independently checked against Drive metadata; only their small indexes were stored locally.
- Remote logger tests followed RED then GREEN (5 passed), including exclusion of tokenizer/checkpoint assets. Full suite:442 passed with14 warnings in7.83s. The initial start log bundles had included tokenizer JSON assets; the filter was corrected and both log-only watchers restarted before further snapshots. Training/evaluation code on these two running VMs was not changed.
- Both recovery relays now pass a release guard requiring a verified final log bundle. A lightweight local supervisor independently verifies final Drive receipts, uploads the small CLI orchestration stream and retries guarded release after successful workload completion. This is a release condition, not a guarantee against an external backend reclaim.
- Historical limitation: raw stdout lost with the first reclaimed VMs was never persisted and cannot be recovered. Recovered first-attempt metadata is preserved separately; no claim of complete historical stdout is made.
- Both current warm evaluations independently reproduce full-dev greedy bounded-CoT Macro-F1 0.5217665228722886, accuracy0.523 and parse1.0. Training is now updating with finite losses/gradients and zero active score replay drift. These are internal development observations, not test or paper results.
- P0R14 readout diagnostic registered on a third G4 before scoring: fixed label-blind256-row dev subset(seed42), same exact greedy reasoning tokens, primary token-greedy answer retained, secondary complete canonical-suffix MAP reported separately. It does not select a checkpoint or alter R13. Four remote RED tests confirmed the diagnostic was missing before its implementation.

- Readout diagnostic completed remote regression:449 passed,14 PyTorch deprecation warnings in7.81s. Added a nonuniform causal-logit indexing oracle and a regression that keeps256-example results in explicit dev_subset files, excluded from the full-dev collector. Read-only independent review found no correctness blocker; the subset accounting caveat was fixed before inference.
- Deployed the logger-only coverage update to both R13 VMs and restarted only their log watchers (PIDs25814/51487); operational stdout now appends across restarts. Model/trainer/evaluator source on those two VMs remains frozen.
- P0R14 is executing through scripts.run on the third G4 with a connected foreground CLI. Its start log bundle was verified by the remote uploader; the connected client requires independent final Drive metadata verification and an uploaded small orchestration log before releasing the third GPU. Source/protocol/checksum records are preserved remotely. No local model/checkpoint/bulk-log archive was downloaded.

- Independently verified both step50 full-log archives against Drive metadata: primary1,306,849bytes/45files; accuracy1,322,333bytes/41files. This includes full raw rollout evidence and runtime logs through each snapshot; archive bytes remain remote.
- P0R14 completed successfully on the fixed256-row dev subset: token-greedy Macro-F1 0.5014176856282119, canonical MAP 0.5092689034975383;4 changed predictions,2 repairs,0 regressions, net2. Both parse1.0. These are paired diagnostic subset results, not full-dev, test, RL gain or a >=0.58 outcome. Few changes give no evidence that suffix readout alone is the principal bottleneck in this warm checkpoint; no full-dev readout sweep is justified by this probe. R13 selection and decoding remain frozen.
- Probe terminal logs and its small local orchestration stream were independently verified on Drive before the probe GPU was released(returncode0). Only the compact summary/index is local; no archive or model bytes were retrieved. Two main RL jobs remain live, latest observed updates52/51.


### Early checkpoint 100 label-only dev readouts (secondary, 2026-10-15)

| Run | Source | Macro-F1 | Accuracy | Parse |
|---|---|---:|---:|---:|
| HN10R13P100L | N10R13 | 0.57419 | 0.574 | 1.000 |
| HN11R13P100L | N11R13 | 0.56982 | 0.569 | 1.000 |

Both cover 1,000 ViANLI dev examples and are label-only secondary views at step 100. Remote metadata archives were independently checked against Drive size and MD5. Checkpoint 100 is not the preregistered primary full-CoT selection; test was not accessed and these figures are not paper results.
