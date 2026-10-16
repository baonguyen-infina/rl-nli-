# Yêu cầu và ranh giới công việc

## Ý định người dùng mới nhất

- Cần phương pháp LLM cho Vietnamese NLI, nền tảng SFT + RL, có lý luận để phát triển bài báo tốt. Mốc hiệu năng mong muốn sau cùng là **test Macro-F10.58**, không phải accuracy.
- Model chính `Qwen/Qwen3.5-2B`, instruction/post-trained; quá lắm3B. Người dùng đã cấm4B về sau. Các YAML4B là lịch sử cấu hình, không được tự chạy.
- **Không teacher model, không distillation, không teacher rationales.** Frozen reference trong RL là bản sao adapter SFT cùng seed để tính KL; đây không phải teacher sinh nhãn/rationale.
- Google Colab CLI, GPU tier **G4**, tối đa3 session song song. Đây là trần, không phải phải dùng đủ3. Phần cuối chỉ còn2 lane được cấp.
- Không ML nặng trên Mac: không load checkpoint, model inference/training, cài stack ML để test locally. Các thao tác code, JSON/CSV, checksum và tài liệu nhẹ được phép.
- Giữ logs đầy đủ: stdout/stderr cả thành công/lỗi, train CSV, rollout, source/config/hash, predictions, metrics và cost. Không tải checkpoints/archives chứa weights về Mac.
- Một seed trước để đánh giá hướng; multi-seed được người dùng hoãn. Không bịa mean±SD hoặc gọi kết quả một seed là robust.
- Báo cáo hiệu năng cho người dùng và paper **chỉ dùng test**. Dev được dùng chọn checkpoint, chẩn đoán và quyết định compute nội bộ. Không dùng training reward làm thước đo thành công.
- Chỉ mở baseline/ablation có ích sau khi phương pháp chính có bằng chứng; có sẵn nhiều đối chứng rồi. Ưu tiên phân tích kết quả hiện có để chọn việc tiếp.
- Chưa có lời hứa hợp lệ về acceptance A/A*/Q1; thứ hạng venue không suy ra từ một ngưỡng F1.

## Spec repo còn hiệu lực

Nguồn chính `AGENTS.md`; có bản chụp trong `snapshot/repo/AGENTS.md`.

1. Không đưa `uid`, `intended_label` hay phần mã nhãn trong UID vào prompt/feature. UID của ViANLI có thể tiết lộ nhãn được yêu cầu viết.
2. `formats/parser.py` là parser chung cho reward/eval. Không lén dùng parser lỏng riêng khi báo cáo.
3. Nhãn chuẩn: entailment, neutral, contradiction. Prompt/premise/hypothesis tiếng Việt.
4. Mỗi experiment qua `python -m scripts.run --run ID --seed S`, output `results/<ID>_<model>_s<S>/`. Run tồn tại không được ghi đè bằng train mới.
5. Hyperparameters chung ở `configs/base.yaml`; lựa chọn và khác biệt run ở `configs/matrix.py`, YAML sinh ra ở `configs/runs/`.
6. Checkpoint chọn bằng full dev Macro-F1; tie trong runner chọn checkpoint muộn hơn. Dev-mini chỉ nội bộ.
7. Test cần manifest khóa config/model/adapter, gateG4 và once-only receipt. Chạy lại lệnh chỉ tái sử dụng output đã hoàn tất hoặc từ chối, không sinh lại để tìm số đẹp.
8. Paper tables/figures do analysis sinh từ test artifacts, không chép tay số vào manuscript.
9. Tests parser/reward/data/statistics trước khi sửa các module này; mọi ML regression/GPU smoke chạy Colab.

## Phần spec cũ đã bị yêu cầu sau thay thế

- AGENTS còn teacher27B, E4/E5 teacher pipeline và A5 gồm4B: **không được kích hoạt trong hướng hiện tại**.
- Ma trận E0–E13/G0–G5 là nguồn gốc dự án; hướng native allocation và label-marginal đã phát triển bằng runID riêng. Không nhầm đó là tất cả đều đã chạy đủ.
- Goal tool cũ dev≥.55/8h không phản ánh mốc test≥.58 về sau. Goal đang blocked do authentication, không phải completed.
- README repo còn hướng dẫn local ML và số test cũ; nó không thay được ràng buộc người dùng mới nhất.

## Dữ liệu

- ViANLI: train8012/dev1000/test1000; HF `uitnlp/ViANLI`. Không có round trong bản HF đang dùng.
- Prepared row có `id,dataset,premise,hypothesis,label,topic,premise_id,intended_label`; metadata phân tích không phải feature.
- Native groups xây theo **nội dung premise chuẩn hóa**, không đọc nhãn/UID để nhóm; nhóm tối đa3. Gold chỉ vào loss và construction của matched random control.
- Bản hash đã dùng cho cohortR18–R20: train SHA256 `4edd41c3aa8a1d9bdd3bac36a3b80f303c7e973e5db5eb764ae285225f90a5c7`; dev `a9dbfa914892e2b636bf11660cb91bd4698f7d4a72df8f879e9adf526fb29389`. Evidence: `evidence/internal/analysis/r18_prepared_data_hashes.json`.
- ViNLI/ViMedNLI có trong định hướng ban đầu và paper hybrid người dùng đưa; chưa biến thành bằng chứng đa benchmark cho phương pháp RL này.

## Tiêu chuẩn nhận một kết quả

Phải phân biệt training complete, checkpoint selection complete, test complete và archive verified. Một file metrics, progressbar100%, config hoặc process parent sống đều chưa chứng minh cả chuỗi đã xong. Việc CLI mất kết nối không tự chứng minh workload đã dừng. Test cũ đã mở nên các follow-up tiếp theo là exploratory; không mô tả như một test hoàn toàn chưa bị quan sát trong suốt nghiên cứu.
