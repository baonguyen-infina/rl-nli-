# Bàn giao toàn bộ session ViANLI SFT + RL

Ngày bàn giao: **16/10/2026**, múi giờ Asia/Ho_Chi_Minh. Repo gốc: `/Users/infina/Documents/rl-nli`.

Đây là điểm bắt đầu cho session mới. Bộ bàn giao gồm giải thích, danh mục run sinh từ code và metadata, bản chụp source/config/test, các ghi chép nghiên cứu, bằng chứng nhỏ và các script vận hành Colab. Không chứa checkpoint, model weights, token đăng nhập hay private key. Các file trong `evidence/internal/` có số dev/training và chỉ dùng nội bộ.

## Trạng thái cần biết ngay

- Mục tiêu người dùng cuối cùng: **ViANLI test Macro-F1 ≥ 0.58**, bằng LLM 2B instruction/post-trained + SFT + RL, không teacher/distillation. Chưa có kết quả đã xác minh đạt mục tiêu này.
- Goal cũ trong công cụ vẫn ghi dev ≥0.55 trong 8 giờ; yêu cầu về sau đã nâng mục tiêu. Không dùng mục tiêu cũ để tuyên bố đã xong.
- Hai kết quả test RL gần nhất được xác minh: **N15R18 = 0.5651534564**, **N16R19 = 0.5606293811**, n=1000, seed=0, parse=1.0. Xem toàn bộ kết quả đã biết trong [05_TEST_RESULTS_AND_PAPER.md](05_TEST_RESULTS_AND_PAPER.md) và `catalog/test_results.json`.
- **N20R20** đã được gửi SIGINT có chủ đích tại bước quan sát 109; kiểm tra ngay sau đó thấy không còn train/eval worker và GPU trống. Checkpoint100 đã backup. Chưa xác minh việc gom logs cuối, parent/test worker thoát và trả session. Không tự chạy lại run này, không test run dừng sớm.
- **N21R20** đã chạy và có checkpoint100; observer checkpoint200 đã được khởi động. Trạng thái hiện tại chưa biết vì mất xác thực Google. Không nói nó vẫn train chỉ dựa vào file trạng thái cũ.
- Google từ chối refresh credential và đòi đăng nhập lại. Các vòng monitor local cũ đã được dừng. **Không xác nhận tất cả GPU đã tắt.**
- Bản ACL draft đã có ở `paper/`, dựa trên cohort test cũ gồm14 evaluation views. Đây là draft nghiên cứu một seed/một benchmark; chưa có cơ sở gọi là paper A/A* đã đạt yêu cầu.

## Thứ tự đọc cho session mới

1. [START_NEXT_SESSION.md](START_NEXT_SESSION.md) — prompt có thể dán nguyên vào session mới.
2. [01_REQUIREMENTS.md](01_REQUIREMENTS.md) — yêu cầu mới nhất và các phần spec cũ đã bị thay thế.
3. [06_CURRENT_STATE_AND_RECOVERY.md](06_CURRENT_STATE_AND_RECOVERY.md) — việc đang dang dở, chỗ cần kiểm tra trước.
4. [02_FRAMEWORK.md](02_FRAMEWORK.md) — kiến trúc, toán, loss, reward và các giới hạn.
5. [03_EXPERIMENT_HISTORY.md](03_EXPERIMENT_HISTORY.md) cùng `catalog/experiment_registry.md` — các hướng đã cấu hình/thử/chạy và bằng chứng.
6. [04_FINDINGS_AND_FAILURES.md](04_FINDINGS_AND_FAILURES.md) — lỗi đã gặp, nguyên nhân biết/chưa biết, điều không nên lặp lại.
7. [07_COLAB_WORKFLOW.md](07_COLAB_WORKFLOW.md) và [08_CODE_AND_EVIDENCE_MAP.md](08_CODE_AND_EVIDENCE_MAP.md) — thao tác và vị trí source/log/backup.
8. [09_RESEARCH_AND_NEXT_DECISIONS.md](09_RESEARCH_AND_NEXT_DECISIONS.md) — prior work, giả thuyết còn mở, tiêu chí chọn việc tiếp.

## Các thư mục

| Đường dẫn | Nội dung |
|---|---|
| `catalog/` | Danh mục tất cả run trong matrix/YAML/result metadata; bảng test; inventory backup và source |
| `evidence/internal/` | Bản sao metadata, diagnostics, ledger, kế hoạch, protocol và CSV nội bộ |
| `evidence/results_metadata/` | Config/cost/metrics/checkpoint selection/train CSV nhỏ; không sao chép raw predictions hay trọng số |
| `snapshot/repo/` | Bản chụp source Python, tests, configs, tài liệu và paper đang có; không coi là source lịch sử của mọi run |
| `operators/` | Bản sao script Colab/monitor/archive trước ở `/private/tmp`; nhiều script có PID/đường dẫn cứng, phải đọc trước khi dùng |
| `MANIFEST.json` | SHA256/kích thước của các file trong bộ bàn giao |
| `VALIDATION.json` | Kết quả kiểm tra bộ bàn giao, đối chiếu test metadata và kiểm tra không chứa weights/credential |

Danh mục phân biệt **có config** với **có bằng chứng chạy**; không biến mọi YAML thành một kết quả thực nghiệm. Snapshot source hiện tại không thay được source hash ghi trong archive của từng run. Bản này tổng hợp những gì còn trong workspace và ghi chép session; không phải bản xuất nguyên văn toàn bộ lịch sử chat.

Đã lập danh mục190 run ID/cấu hình và16 kết quả test đã biết.14 test views có predictions local được tính lại metric và đối chiếu receipt/checksum;2 kết quả N15/N16 dựa trên recorded remote verification cùng final archive receipts. Không query lại remote hay chạy model trong lần bàn giao. Có thể tái tạo các bảng/inventory bằng `.venv/bin/python session_handoff/2026-10-16/build_handoff.py`; helper chỉ đọc code/metadata, không gọi mạng hoặc import thư viện ML.

Không xóa hay reset các thay đổi cũ trong repo: workspace có rất nhiều file sửa và chưa commit. Bản chụp `catalog/git_status.txt` ghi lại trạng thái lúc bàn giao.

Để tiết kiệm context, session mới chỉ cần đọc các Markdown tóm tắt ở tầng này trước. Source và evidence mở theo câu hỏi cụ thể; không dump toàn bộ hơn1.500 file vào prompt. Những file raw lớn đã có trong workspace/remote archives không cần đọc lại nếu checksum và receipt đủ cho thao tác đang làm.
