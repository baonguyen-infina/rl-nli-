# Prompt để bắt đầu session mới

Copy nội dung dưới đây vào session mới mở tại `/Users/infina/Documents/rl-nli`:

```text
Đọc session_handoff/2026-10-16/README.md, 01_REQUIREMENTS.md,
06_CURRENT_STATE_AND_RECOVERY.md rồi các Markdown tóm tắt01–09 ở tầng này.
Source/evidence chỉ mở khi cần, không dump toàn bộ folder vào context.
Đây là context bàn giao của dự án ViANLI SFT + RL, không phải yêu cầu viết lại từ đầu.

Mục tiêu cuối: test Macro-F1 >=0.58 với Qwen3.5-2B instruction/post-trained,
SFT + RL, không teacher/distillation, tuyệt đối không4B. Tối đa3 Colab G4.
Không load weights, chạy ML, train, infer hay ML test trên Mac.
Chỉ báo cáo benchmark scores từ test; dev/training là nội bộ.
Đọc bằng chứng cũ trước khi đề xuất/chạy thêm. Không lãng phí compute bằng baseline
hoặc biến thể mới thiếu giả thuyết và đối chứng rõ ràng.

Đầu tiên khôi phục xác thực Colab nếu cần, rồi read-only kiểm tra assignment/processes,
cost/log/test claim của N20R20 và N21R20. Các status/PID trong handoff đã cũ.
N20 đã chủ động dừng bước109, không tự restart và không mở test.
Checkpoint100 của cả2 đã backup private GitHub draft; phải kiểm tra logs cuối
và archive trước khi trả bất kỳ GPU nào.
N21 có quy tắc xét checkpoint200 bằng dev, nhưng controller chưa xác minh/thực thi
vì mất xác thực; observer không có chức năng tự dừng.
Nếu pipeline/test đã hoàn tất, thu kết quả đúng một lần và ghi rõ việc gate chưa được thực thi.
Không chạy lại test hay đổi checkpoint dựa vào test.

Chưa có test >=0.58 đã xác minh. N15R18 test MF1 .5651534564;
N16R19 .5606293811. Bản paper ACL cũ có14 test views; chưa chứng minh RL có lợi
hơn SFT hoặc policy-off control, chưa đủ chứng cứ A/A*.
Mọi tuyên bố mới phải bám metadata/prediction/hash thật.
Giữ toàn bộ log/checkpoint trên Colab hoặc archive private;
chỉ lưu code và metadata nhẹ trên máy.

Sau khi xác minh trạng thái thực tế, tóm tắt kết luận và chọn bước tiếp theo dựa trên
dev diagnostics cùng prior work. Đừng lặp lại các hướng đã thất bại chỉ bằng đổi tên.
```

Nếu chỉ cần tiếp nhận context mà chưa chạy thêm, không gọi GPU hoặc OAuth. Nếu cần thực nghiệm, mã OAuth/link từ session cũ có thể hết hạn; tạo một flow mới và giữ process nhận mã tương ứng. Không tái dùng mã Google cũ trong lịch sử chat.
