# ViANLI-RL: SFT + RL với Suy Luận Căn Chỉnh Tường Minh cho ViANLI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baonguyen-infina/rl-nli-/blob/master/notebooks/g0_setup.ipynb)

Mã nguồn thực hiện spec nghiên cứu: **SFT + RL với suy luận căn chỉnh tường minh cho bài toán suy luận tự nhiên tiếng Việt trên bộ dữ liệu ViANLI (Adversarial NLI)**.

## 🚀 Chạy Ngay Trên Google Colab
Bấm vào nút **Open In Colab** ở trên hoặc mở file [`notebooks/g0_setup.ipynb`](notebooks/g0_setup.ipynb).

Tất cả các bước từ thiết lập môi trường, tải dữ liệu, unit tests, chạy Cổng G0 (GRPO smoke test) và Giai đoạn G1 (Zero-shot E0..E2, SFT E3, CafeBERT B1, Hyp-probe) đều được cấu hình sẵn dạng one-click.

---

## 🛠️ Cài Đặt Cục Bộ (Local)

```bash
git clone https://github.com/baonguyen-infina/rl-nli-.git
cd rl-nli-

# Khởi tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Chuẩn bị dữ liệu ViANLI & sinh configs ma trận
python -m data.prepare
python -m configs.matrix

# Chạy unit tests (76 tests CPU)
python -m pytest -q
```

---

## 📋 Cấu Trúc Mã Nguồn

```text
rl-nli/
├── common/             # Nhãn, xử lý chuỗi NFC/casefold, đọc ghi file an toàn
├── formats/            # Parser hợp nhất, natural logic alignment, prompt templates
├── rewards/            # Hàm thưởng R = r_acc + 0.2*r_fmt + 0.3*r_acc*r_struct (TRL)
├── data/               # Chuẩn bị ViANLI (8,012 train, 1,000 dev, 300 dev-mini), hyp-probe
├── configs/            # base.yaml và matrix.py sinh configs cho từng run
├── eval/               # Greedy vLLM evaluation, macro F1, records
├── train/              # SFTTrainer, GRPOTrainer, CafeBERT fine-tuning
├── scripts/            # run.py (entrypoint duy nhất), check_g1.py (kiểm tra cổng G1)
├── notebooks/          # g0_setup.ipynb (launcher cho Google Colab / Modal)
└── tests/              # 76 tests kiểm tra tự động trước khi chạy GPU
```

---

## 🎯 Quy Tắc Nghiên Cứu Bất Di Bất Dịch
1. **Không đưa `uid` vào prompt hay feature** (tránh leak nhãn annotator ViANLI).
2. **Không chạm test split trước giai đoạn G4** (khóa kiểm tra nghiêm ngặt `allow_test=True` + `G4_UNLOCK=1`).
3. **Một parser duy nhất** (`formats/parser.py`) dùng chung cho cả reward, evaluation và filtering.
4. **Mọi run đi qua `scripts/run.py --run <ID> --seed <S>`** và ghi vào `results/<run_id>_<model>_s<seed>/`.
