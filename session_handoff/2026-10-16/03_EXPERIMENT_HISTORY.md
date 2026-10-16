# Những gì đã thử và các lần đổi hướng

Đọc kèm `catalog/experiment_registry.md`/JSON/CSV. Danh mục sinh từ matrix, YAML và **metadata thực sự còn lại**, không suy diễn rằng có config nghĩa là đã chạy. Các số dev ở evidence là nội bộ; không đưa vào paper.

## Diễn tiến phương pháp

| Giai đoạn | Run/family | Ý tưởng, lý do thử và kết luận còn giữ |
|---|---|---|
| Khởi đầu | E0–E13, D3, D9–D26, G0 variants | Label/CoT/explicit alignment, ordinary GRPO/GTA, balanced schema examples, flat fallback, rule supervision. Parser/reward/quote-binding đã được làm rõ; người dùng cấm teacher về sau nên gold-only label SFT trở thành warm start. Một số row chỉ là cấu hình hoặc đề xuất. |
| Native contrast pilot | N0/N1/N2/N3, D3R/D3R3, R4 | Xây nhóm sibling hypotheses từ premise, conditional gold-label allocation; ordinary/CE/random controls. Điều tra detached scoring, dropout/reference và inference repeats. |
| Allocation channels | N4R4/N5R4, N1R5/N4R5/N7R5 | Tách group reward với direct-score gradient; thêm token credit. Không mặc định mọi lợi ích đến từ RL khi có auxiliary supervised gradient. |
| Context specificity | N5R6/N8R6/N9R6 | Joint allocation, swapped traces và swap+saliency; tồn tại các validation views A/B/L. |
| Longer horizon | N1R7/N4R7/N5R7 | Tăng shared horizon để kiểm tra thiếu exposure; không tune riêng từng cell. |
| Capacity idea | D4R8/N4R8/G0N4R8/V4R8* | Config4B lịch sử, comment trong matrix nói unlaunched; người dùng cấm4B. Không phục hồi hướng này. |
| Audited2B warm start | D3R9, N1R9/N4R9/N4W9 | SFT full_budget và full_reasoning400; ordinary/native/old warm contrast. Fixed-padding deterministic scoring; greedy vLLM repeat variance còn đáng kể. |
| Joint and bounded reward | N5R10/N6R11/N0R11 | Joint logQ, bounded correctness-gated bonus, decisive policy-off allocation control. Cả CE/direct terms khiến cần đối chứng policy-off. |
| Random grouping | N7R12 | Gated joint với random blocks, stop200 trên LR horizon400; matched label-slot exposure. Label-only checkpoint200 view là view riêng, không relabel primary CoT selection. |
| Frozen test cohort + ACL draft | H...L/H...C,14 views | Khóa evaluator Transformers single-sequence, source/config/adapter rồi chạy test. Tất cả số ở paper draft lấy từ cohort này. Không có primary≥.58; policy-off CoT control cao hơn một số RL rows. |
| Finite-label utility | N10R13/N11R13 | Macro-F1 marginal utility versus accuracy utility; bỏ native assignment objective, mask sampled answers, RLOO giữa reasoning samples. Hai attempt do infrastructure interruption, không phải hai seed. |
| Readout probe | P0R14 | Greedy so với canonical-MAP trên256dev samples. Diagnostic only, không chọn checkpoint, không phải test result/new training method. |
| Full random budget | N12R15 | Fresh400 từ D3R9, giữ gated_joint/random; trainCoT/evalgreedylabel. Không phải optimizer resume từ N7R12. Đạt train400 nhưng selection bị gián đoạn trong đợt user stop; checkpoint200 được cứu và đọc riêng. |
| Trace interventions | N13R16/N14R17 | Unbounded swap và swap+own-allocation saliency. Có wrong-positive scalar policy advantages; dropout/replay audit vẫn ổn. User stop khiến endpoint/selection chưa hoàn chỉnh. |
| Bounded swap + geometry repair | N15R18 | Fresh400 gated_trace_swap/random, source mới sửa padding geometry trên original/rotated/saliency. Đã xong dev selection, once-only test và archive. |
| Matched grouping | N16R19 | Cùng bounded swap400 nhưng native grouping. Exact scheduled examples/label templates giữ matched. Đã xong test/archive. Không phải ablation chỉ riêng trace swapping. |
| Full saliency revisit | N20R20 | Fresh400 trace_saliency, smoke passed; checkpoint100 eval và backup verified. Chủ động SIGINT tại observed109 vì gain chưa thuyết phục so với lane nhanh hơn và unrestricted credit issue còn đó. Chưa final archive/release; khôngtest. |
| Full label utility revisit | N21R20 | Fresh400 marginal_f1, cùng warm/readout/horizon; checkpoint100 có, checkpoint200 observer chạy. Quy tắc futility được ghi trước200 nhưng **chưa xác minh/thực thi** khi mất auth. Hiện trạng remote chưa biết. |

## Lịch sử vận hành quan trọng

- E8 log người dùng đưa: train500/500 đã kết thúc nhưng eval vLLM thất bại do free GPU71.35GiB < requested80.73GiB ở utilization0.85. Không có test E8 hợp lệ từ traceback đó. Runner về sau tách training child và inference process để giải phóng GPU.
- E5 người dùng đưa Accuracy.3000/MF1.2761/parse.8710; không có split/provenance đủ trong hội thoại để coi là verified test headline. Dữ liệu đó chỉ là bối cảnh thất bại ban đầu.
- Có nhiều lần user yêu cầu stop toàn bộ session khi đi làm/tắt máy, rồi cho resume. Không coi stopped/partial run là completed; không lấy bước train cao nhất làm test score.
- R13 attempt1: archive cuối biết N10 tới60, N11 tới20; không có checkpoint100. Attempt2 cùng frozen seed/config là recovery, không phải independent replicate. N11 attempt2 có verified dev output checkpoint100; N10 status từng thấy train200 nhưng CLI exit1, cần evidence/archive để phân giải final trạng thái.
- R15/R16/R17 old cohort được xác nhận terminated trước khi cấp recovery G4R18. `HN12R15P200L` và `HN13R16P100L` là explicit checkpoint readouts của interrupted parents.
- Drive gần hết quota và project creation vướng Google Cloud terms. Đã chuyển durable checkpoint/log backup sang **owned private GitHub draft release**. Không công bố dữ liệu hay model.
- R18/R19 hoàn thành trên2G4, sau đó tái sử dụng cùng session names để chạy N20/N21. **Tên session không đồng nghĩa runID hiện tại.**

## Đọc trạng thái đúng

`G0*` là smoke/gate, `V*` là validation/repeat, `H*L/C` thường là frozen inference view của parent adapter, `P0R14` là diagnostic. Không cộng chúng thành số mô hình train độc lập. Các bảng CSV cũ có thể không chứa latest remote-only N15–N21; registry ghi source và mức xác minh riêng.

Chưa có nhật ký nguyên văn đầy đủ cho mọi thao tác từ đầu cuộc trò chuyện; các raw stdout/rollouts có trong remote archives theo receipts. Bộ bàn giao giữ tất cả source/config và metadata nhẹ hiện còn trong workspace, cùng timeline có căn cứ trên.
