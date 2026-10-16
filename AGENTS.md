# AGENTS.md — hướng dẫn cho AI viết code

Repo này thực hiện spec "SFT+RL với suy luận căn chỉnh tường minh cho ViANLI".
Spec đầy đủ (lý do, RQ, ma trận, phân tích): https://claude.ai/code/artifact/c73052e7-feb0-47b0-b9e5-4fc13b09ff96
Mỗi file .py hiện chỉ có docstring nói nó phải làm gì. Implement theo đúng docstring + các quy tắc dưới đây.

## Quy tắc bất di bất dịch
1. **Không đưa `uid` vào prompt hay feature.** Trường thứ 5 của uid ViANLI (11/21/31) là nhãn annotator được yêu cầu viết — gần như là đáp án.
2. **Không chạm test split trước giai đoạn G4.** Chọn checkpoint bằng dev; dev-mini chỉ để vẽ đường cong nội bộ. **Mọi kết quả thực nghiệm ghi trong paper (bảng, hình, abstract, discussion) chỉ lấy từ tập test.** Số dev và training log chỉ lưu nội bộ; không dùng thay số test khi chưa mở G4.
3. **Một parser duy nhất** (`formats/parser.py`) cho reward, eval và teacher filter.
4. **Mọi run đi qua `scripts/run.py --run <ID> --seed <S>`** và ghi vào `results/<run_id>_<model>_s<seed>/`. Không chép số tay vào paper; bảng/hình sinh từ `analysis/`.
5. Không tune riêng từng ô. Hyperparameter nằm ở `configs/base.yaml`; khác biệt giữa các run chỉ ở `configs/matrix.py`.
6. Viết test trước cho `formats/`, `rewards/`, `data/prepare.py`, `analysis/stats.py` (chạy được không cần GPU).

## Hợp đồng dữ liệu
- Nhãn: `entailment | neutral | contradiction` (tiếng Anh trong output LLM). Prompt và premise/hypothesis tiếng Việt.
- ViANLI (HF `uitnlp/ViANLI`, file `vianli_{train,dev,test}.jsonl`, cột `uid, premise, hypothesis, label`): 8,012 / 1,000 / 1,000. uid = `uit_Adver_<premise_id>_<k>_<intended>_<topic>`; topic 01–13 (bảng mã ở `data/topics.py`, khớp Table 3 paper ViANLI). Bản HF không có round.
- ViNLI: bỏ nhãn OTHER. ViMedNLI: bản local của người dùng.
- Mỗi dòng sau prepare: `id, dataset, premise, hypothesis, label, topic, premise_id, intended_label` (intended_label chỉ để phân tích).

## Format output
```
label : <answer>neutral</answer>
cot   : <think>...</think><answer>...</answer>
align : <align>
        H1: "<đoạn hypothesis>" <-> "<đoạn premise>" | match
        H2: "<đoạn hypothesis>" <-> ∅ | missing
        H3: "<đoạn hypothesis>" <-> "<đoạn premise>" | conflict
        </align><answer>...</answer>
GTA   : thêm <guess>nhãn</guess> ở đầu (chỉ cho cot/align)
```
Align: 1–6 dòng; `missing` phải đi với ∅. Luật: có conflict → contradiction; không thì có missing → neutral; còn lại → entailment.

## Reward
`R = r_acc + 0.2*r_fmt + 0.3*r_acc*r_struct`; r_struct = 1 khi mọi đoạn là chuỗi con (sau NFC, gộp khoảng trắng, casefold), đoạn H phủ ≥ 50% âm tiết hypothesis, và luật cho đúng nhãn trong <answer>.
TRL reward func nhận `(prompts, completions, **kwargs)`; completion có thể là chuỗi hoặc `[{"role":"assistant","content":...}]`.

## Model & thư viện
- Chính `Qwen/Qwen3.5-2B` (non-thinking mặc định; model đa phương thức, kiến trúc lai Gated DeltaNet + attention → chỉ nạp phần ngôn ngữ, vLLM `--language-model-only`). A5: `Qwen3.5-0.8B`, `Qwen3.5-4B`. Teacher `Qwen3.5-27B`.
- `trl==1.15.0`. Điểm móc trong GRPOTrainer: `_generate_and_score_completions` trả dict có `advantages` (+HypW nhân ở đây) và `tool_mask` (GTA đặt 0 trên token guess; `_compute_loss` nhân completion_mask với tool_mask).
- Rollout: temperature 1.0, top_p 1.0, không top_k/presence penalty. Eval: greedy.

## Ma trận (run_id)
| Recipe \ Format | label | cot | align |
|---|---|---|---|
| ZS | E0 | E1 | E2 |
| SFT | E3 | E4 | E5 |
| GRPO | E6 | E7 | E8 |
| SFT→GRPO | E9 (P2) | E10 (từ E4) | **E11** (từ E5) |
| GTA | – | E12 | E13 |

Encoder: B1 CafeBERT, B2 GRIP, B3 số trích. Ablation: A1 E11 bỏ r_struct · A2 +HypW · A3 GTA tắt luật cos · A4 bỏ mẫu rationalized · A5 quy mô · A6 Llama-3.2-3B. P1 = 3 seed (0,1,2).

## Thứ tự làm (cổng phải qua mới đi tiếp)
- G0: môi trường + `data/prepare.py` + test parser/reward → một bước GRPO chạy hết.
- G1: E0–E2, E3, B1, hyp probe → E3 > ZS trên dev, parse ≥ 99%.
- G2: teacher + E4/E5 + pilot E11 200 bước → reward tăng, parse ≥ 95%, đọc 50 rollout.
- G3: lưới P1 seed 0 → khoá config.
- G4: seed 1–2, P2, test một lần, phân tích.
- G5: `analysis/make_tables.py`, `make_figs.py` → `paper/`.

## Đầu ra mỗi run
`config.yaml`, `train_log.csv`, `predictions_<split>.jsonl` (`id, dataset, topic, gold, pred, guess, parse_ok, rule_label, grounded, coverage, n_tokens, raw_output`), `metrics_<split>.json`, `cost.json`.

Không chắc điều gì → hỏi người, đừng tự đổi spec.
