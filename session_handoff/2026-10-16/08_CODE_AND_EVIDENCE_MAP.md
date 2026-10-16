# Bản đồ source và bằng chứng

## Source chính

| File/module | Vai trò |
|---|---|
| `configs/base.yaml`, `configs/matrix.py`, `configs/runs/*.yaml` | Shared settings, run matrix, generated overrides |
| `common/io.py` | Paths, merge config/profiles, JSONL/CSV IO; load_config là cách lấy effectiveprofile |
| `data/prepare.py`, `data/topics.py` | Split gates, normalized rows, UIDmetadata extraction; topic chỉ phân tích |
| `formats/parser.py`, `formats/prompts.py` | Parser/output contract và Vietnamese prompts |
| `rewards/` | Legacy accuracy/format/structuralreward wrappers |
| `train/common.py`, `train/sft.py` | Model/LoRA loader, SFT masks/templates, vLLMweight sync/compatibility helpers |
| `train/grpo.py`, `train/gta.py`, `train/rule_supervision.py` | Legacy alignment/GTA/rule supervision |
| `train/native_grpo.py` | Actual nativeTRLtrainer hooks; rewards, cache/replay, loss, audits, sampler, train orchestration |
| `train/native_groups.py` | Native/random blocks, labelmultiset assignment, logQ and tensorvariant |
| `train/native_allocation.py` | Detached exact scoreJacobian cache and microbatch surrogate |
| `train/native_reward.py` | Correctness-gated boundedQ/swap bonus, policyoff coefficient |
| `train/native_trace_swap.py`, `train/native_saliency.py` | Donor indices, gradientactivation attribution and tokenweights |
| `train/label_marginal.py` | Confusion utility, finite-label expectation, RLOO advantages |
| `train/native_training.py`, `train/native_rollout.py` | CE prefixes, masks, deterministic scoring, boundedgeneration and samplerindexing |
| `train/audit.py` | Live CSV/callback evidence |
| `scripts/run.py`, `scripts/train_worker.py` | Entrypoint, frozenrun, childtraining, checkpointselection, once-onlytest |
| `eval/hf_generate.py`, `eval/predict.py` | FrozenHF/vLLM greedyreadout, config/runtime/provenance |
| `eval/final_protocol.py`, `eval/final_predict.py` | Artifact locking, protocol validation, claims/receipts and finaltest execution |
| `eval/metrics.py`, `eval/records.py` | Unified records, n/acc/MF1/parse/perclass/confusion |
| `scripts/remote_logs.py`, `scripts/remote_artifacts.py` | Logcollection and archivefile selection; omitcredentials/weights selectively |
| `analysis/collect.py`, `paper_report.py`, `stats.py` | Completed-artifact collection, test-onlytables/paired uncertainty |
| `tests/` | Parser/data/reward/mask/gradient/replay/exposure/runtime/testguard/archive/reporting regressions |

Snapshot source trong `snapshot/repo/` là localtree ngàybàn giao. Runhistorical có source riêng trong archive và hashreceipt. Không overwrite repo bằng snapshot để “đồng bộ” một cáchmùquáng.

## Bằng chứng nội bộ quan trọng

| Evidence gốc dưới `artifacts/analysis/` | Dùng để trả lời |
|---|---|
| `r13_ledger.md`, `r13_research_boundary.md` | Labelmarginal design, attempt history, gates và noveltylimits |
| `r13_attempt1_summary.json` | Partial interrupted N10/N11, checkpoints chưacó |
| `r13_attempt2_accuracy_verified_result.json` | N11 verifieddev checkpoint100; không phải test |
| `r14_readout_summary.json` | Diagnostic256 devsubset, không đượcselection |
| `r15_ledger.md`, `r16_r17_review.md` | Fullrandom/swap/saliency progression, credit inversion, loggingfix |
| `r9_fixed_runtime_repeat_audit.json` | Fixedcheckpointgreedy repeats không đồngnhất |
| `r18_remote_status.json`, `r19_remote_status.json` | Remote verification của completed N15/N16 test (timestamp cũ) |
| `github_backup_receipts/N15R18_final.json`, `N16R19_final.json` | Durablefinalarchive SHA/size/assetIDs |
| `r18_training_credit_diagnostic.json`, `r19_training_credit_diagnostic.json` | Bounded scalarcredit observations |
| `r20_ledger.md`, `r20_paired_dev_diagnostic.json` | Revised bottleneck diagnosis, repairs/regressions |
| `r20_n20_checkpoint100_internal.json`, `r20_n21_checkpoint100_internal.json` | Current100 paireddev, predictionsSHA |
| `r20_n20_training_credit_diagnostic.json` | Unrestricted currentcreditcounts |
| `r20_literature_check.md` | Recent priorwork và claimboundaries |
| `r20_auth_recovery_status.json` | Authenticationblocker, retiredlocaloperators, pendingrecovery |
| `native_r20_live_status.json` | Lastpoll errors; khôngphải currentremotestate |

Những file này được copy/redact vào `evidence/internal/analysis/`. Protocols vào `evidence/internal/protocols/`; design/plans vào `evidence/internal/docs/`. Bản sao đã loại secret/signedcapability nếu có; `catalog/copy_inventory.json` chỉraorigin/hash/redaction.

## Kết cấu output mỗi run

```text
results/ID_Qwen3.5-2B_s0/
  config.yaml                   frozen effectiveconfig
  cost.json                     train/eval status
  train_log.csv                 scalars/internalonly
  rollouts_train.jsonl          captured traces and policycredit; remote fullarchive
  initial_eval/                 warmstart sameprotocoldev
  grpo_ckpts/checkpoint-N/       adapters/trainerstate; weights stayremote
  ckpt_eval/checkpoint-N/        greedy full-dev metrics/predictions
  ckpt_scores.json               chosen full-devcheckpoint
  best_adapter/                  selectedadapter
  gradient_smoke.json            realgradientchannels
  reference_audit.json           frozen ref fingerprint
  dropout_audit.json             active dropoutcheck
  group_audit.json               exposure/grouping
  source_snapshot.json           sourcehashes
  execution_provenance.json      environment/method/data scope
  locked_test_protocol.json      final manifest
  .test_claim/receipt.json        once-only executionstatus
  predictions_test.jsonl         savedoutputs; donot regenerate
  metrics_test.json
  cost_test.json
  test_verification.json
  github_backup/*.receipt.json
```

Localmetadata copy không gồm modelweights, rawpredictionJSONL hayfullrolloutJSONL. Các rawfiles đãcó vẫn ở repo/results hoặc archive; inventory giữ vị trí/hash/size để tra. Không suy rằng chúng bịxóa vì handoffkhôngcopy.
