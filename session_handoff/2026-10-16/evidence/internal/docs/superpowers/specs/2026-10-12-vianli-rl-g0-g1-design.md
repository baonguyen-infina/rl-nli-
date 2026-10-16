# ViANLI SFT+RL — Design Spec cho G0–G1

Ngày: 2026-10-12. Nguồn ràng buộc: `AGENTS.md` và docstring của từng module. Bản spec gốc (artifact claude.ai) không truy cập được; mọi giá trị đánh dấu **[đề xuất]** cần người dùng duyệt.

## 1. Phạm vi

**Làm:** môi trường; `common/`, `data/prepare.py`, `data/topics.py`, `formats/*`, `rewards/*`, `analysis/stats.py` (test-first, chạy được không cần GPU); train SFT (E3) và 1 bước GRPO; `eval/predict.py`, `eval/metrics.py`; `scripts/run.py`, `configs/matrix.py` cho E0–E3, B1; `data/hyp_probe.py` (chỉ cross-fitting).

**Không làm:** teacher, E4 trở đi, GTA, ablation, `analysis/make_*`, `eval/intervene.py`, ViNLI, ViMedNLI. Các file này giữ nguyên stub.

**Dữ liệu:** chỉ ViANLI trong giai đoạn này.

## 2. Môi trường

- Phương án A: repo là nguồn duy nhất, notebook chỉ là lớp mỏng. Có thể thêm `modal_app.py` mỏng ở giai đoạn sau.
- **Mac (dev):** `pytest`, `numpy`, `scipy`, `pyyaml`, `datasets`. Test của `formats/`, `rewards/`, `prepare`, `stats` phải pass không cần torch hay vLLM.
- **Colab/Modal:** GPU 96GB (Blackwell). `trl==1.15.0`; `vllm`, `transformers`, `peft` pin ở G0 theo bản đầu tiên chạy được Qwen3.5-2B (có thể cần CUDA ≥ 12.8).
- `artifacts/` và `results/` nằm trên Drive hoặc Modal Volume, symlink vào repo.
- `notebooks/g0_setup.ipynb`: cài đặt, kiểm tra GPU/vLLM, gọi `scripts/run.py`. Không chứa logic.
- `torch`, `vllm`, `trl` chỉ import lazy trong `train/`, `eval/predict.py`, `scripts/run.py`.

## 3. Dữ liệu

### `data/prepare.py`
- Tải `uitnlp/ViANLI` (`vianli_{train,dev,test}.jsonl`); assert 8,012 / 1,000 / 1,000.
- Regex uid `uit_Adver_<premise_id>_<k>_<intended>_<topic>`; `intended` ∈ {11,21,31} = entailment, neutral, contradiction; `topic` 01–13. Không khớp thì raise.
- Schema: `id, dataset, premise, hypothesis, label, topic, premise_id, intended_label`; `dataset` = `vianli`.
- dev-mini: 300 mẫu phân tầng theo nhãn, seed 42, lưu `artifacts/data/dev_mini_ids.json`.
- `uid` và `intended_label` không vào prompt hay feature; có test kiểm prompt không chứa `uit_Adver`.
- Test split chỉ đọc qua hàm có `allow_test=True`, yêu cầu biến môi trường `G4_UNLOCK`.

### `data/topics.py`
Bảng 01–13. Bản đầu dùng mã `T01`…`T13`, kèm TODO để điền tên theo Table 3; test chỉ kiểm đủ 13 khóa.

## 4. Parser, align, reward

### `formats/parser.py`
`parse(text, fmt, guess=False) -> Parsed(label, guess, reasoning, align_lines, format_ok)`. Chặt: mỗi tag đúng 1 lần, đúng thứ tự, không text thừa ngoài tag, `<answer>` là 1 trong 3 nhãn tiếng Anh, `<guess>` (nếu bật) đứng đầu. Lỗi thì `format_ok=False`, không sửa đoán. Parser duy nhất cho reward, eval, teacher filter.

### `formats/align.py`
- Dòng: `H<i>: "<đoạn H>" <-> "<đoạn P>" | match|missing|conflict`, hoặc `<-> ∅ | missing`.
- 1–6 dòng; `missing` bắt buộc đi với `∅`; `match`/`conflict` không đi với `∅`.
- Luật: có conflict → contradiction; không thì có missing → neutral; còn lại → entailment.
- Grounding: chuỗi con sau NFC, gộp khoảng trắng, casefold (H tìm trong hypothesis, P tìm trong premise).
- Coverage: tổng âm tiết các đoạn H / âm tiết hypothesis ≥ 0.5; âm tiết = tách theo khoảng trắng trên chuỗi đã chuẩn hóa.
- `struct_ok` = grounding đúng, coverage đạt, nhãn theo luật khớp `<answer>`.

### `rewards/`
- `core.py` (hàm thuần): `r_acc` 0/1, `r_fmt` 0/1 theo `format_ok`, `r_struct` 0/1 theo `struct_ok` (bằng 0 với format `label` và `cot`).
- `R = r_acc + 0.2·r_fmt + 0.3·r_acc·r_struct`.
- `trl_rewards.py`: `r_acc`, `r_fmt`, `r_acc_struct` với chữ ký `(prompts, completions, **kwargs)`; completion là chuỗi hoặc `[{"role":"assistant","content":...}]`; gold, premise, hypothesis lấy từ `kwargs`; trọng số 1.0/0.2/0.3 ở `reward_weights`.

### Test (viết trước, chạy trên Mac)
Output hợp lệ; thiếu tag; tag lặp; text thừa; nhãn lạ; `∅` sai chỗ; đoạn không phải chuỗi con; coverage 0.49 và 0.5; reward với chuỗi và message list.

## 5. Train, eval, matrix

- `formats/prompts.py`: prompt tiếng Việt cho `label`/`cot`/`align`, có/không `<guess>`; mỗi prompt một ví dụ format tự viết; chỉ nhận `premise` và `hypothesis`; chat template Qwen3.5 non-thinking.
- `train/common.py`: nạp phần ngôn ngữ của Qwen3.5-2B (bf16); LoRA trên `q,k,v,o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate,up,down_proj`.
- `train/sft.py` (E3, completion `<answer>nhãn</answer>`, loss chỉ trên completion). **[đề xuất]** LoRA r=32, alpha=64, dropout 0.05; lr 2e-4 cosine, warmup 3%; 3 epoch; batch hiệu dụng 32; `max_len` 1024; chọn checkpoint theo dev accuracy.
- `train/grpo.py`: G0 chạy 1 bước với format `label`. **[đề xuất]** lr 1e-5, `num_generations` 8, `max_steps` 500 (G0: 1), `beta` 0, `loss_type` dapo, rollout T=1.0 top_p=1.0. HypW và hook `_generate_and_score_completions` để giai đoạn sau.
- `eval/predict.py`: greedy, vLLM `--language-model-only`; ghi `predictions_<split>.jsonl` (`id, dataset, topic, gold, pred, guess, parse_ok, rule_label, grounded, coverage, n_tokens, raw_output`; trường không áp dụng = null) và `metrics_<split>.json`.
- `eval/metrics.py`: accuracy, macro-F1, confusion matrix, accuracy theo chủ đề, tỉ lệ parse.
- `configs/matrix.py` định nghĩa E0–E3, B1 và sinh `configs/runs/<id>.yaml` từ `base.yaml`; `scripts/run.py --run <ID> --seed <S>` đọc config, gọi trainer, predict trên dev.
- Mỗi run ghi `results/<run_id>_<model>_s<seed>/`: `config.yaml`, `train_log.csv`, `predictions_*.jsonl`, `metrics_*.json`, `cost.json`.
- B1: fine-tune CafeBERT (mặc định `uitnlp/CafeBERT`, kiểm lại khi chạy) trên ViANLI train, đánh giá trên dev.
- Hyp probe: CafeBERT hypothesis-only, 5-fold cross-fitting trên train lấy `p_hyp`. Gán nhãn hyp-easy/hyp-hard cho test chờ G4.

## 6. Cổng

- **G0:** môi trường chạy được, `prepare` pass, test parser/reward pass, 1 bước GRPO chạy hết.
- **G1:** E0, E1, E2, E3, B1, hyp probe chạy xong; **E3 > ZS trên dev và tỉ lệ parse ≥ 99%**. Script nhỏ tự kiểm và in PASS/FAIL kèm số liệu.

## 7. Ràng buộc bất di bất dịch (từ AGENTS.md)

Không `uid` vào prompt/feature; không chạm test trước G4; một parser duy nhất; mọi run qua `scripts/run.py`; không chép số tay vào paper; không tune riêng từng ô; hyperparameter ở `configs/base.yaml`, khác biệt giữa run chỉ ở `configs/matrix.py`.

## 8. Điểm mở

- Tên chủ đề thật theo Table 3.
- Giá trị đánh dấu **[đề xuất]** cần duyệt.
- Phiên bản `vllm`/`transformers`/`peft` pin lúc G0 trên GPU thật.
- ID Hugging Face chính xác của CafeBERT.
