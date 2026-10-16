# Những phát hiện đã có và các nhận định cần sửa

## Điểm nghẽn có bằng chứng

1. **Reward tăng không chứng minh test tốt hơn.** Nhiều objectives gồm policy +gold CE+direct score term; cần policy-off matched control để quy gain cho RL. Kết quả test cũ chưa chứng minh RL vượt SFT/auxiliary controls.
2. **Greedy inference repeat variance từng lớn.** `r9_fixed_runtime_repeat_audit.json` ghi cùng adapter/config chỉ khác runID vẫn50/1000 predictions và431 raw outputs đổi. Không chọn repeat cao nhất làm headline. Vì vậy cohort test khóa một shared Transformers single-sequence greedy evaluator. Không tuyên bố đã sửa toàn bộ vLLM/Gated DeltaNet reproducibility.
3. **Unbounded trace contrast có credit ngược correctness.** Audit first20steps: N13R16 có39 wrong-positive và109 correct-negative scalar advantages; N14R17 có36/94. N20R20 có36 wrong-positive và101 correct-negative. Đây là hệ quả objective có thể xảy ra, không mặc định bug implementation hay bằng chứng semantics xấu.
4. **Bounded gating bảo vệ raw reward ordering tốt hơn.** N15 first20steps recorded0 wrong-positive/0 correct-negative trong mixed groups. Test vẫn chưa lên.58; correctness-gated reward không phải lời giải đã đủ.
5. **Neutral↔contradiction trade-off nổi bật trên paired dev.** N15 repairs38/regressions28, trong neutral repairs6/regressions16. N16 repairs42/regressions30, neutral3/21. N20 checkpoint100 neutral3/18; N21 checkpoint100 neutral2/19. Đây là descriptive selection data, không phải test analysis hay causal mechanism đã chứng minh.
6. **Nhiều hard errors giữ nguyên.** N15/N16 khoảng391–395 dev rows cùng sai trước/sau; thay reward mà không đo failure mechanism có thể ít ích lợi.

## Nhận định trước đây quá mạnh — không mang sang session mới

- “Accuracy versus Macro-F1 mismatch là mấu chốt”: số liệu trên benchmark khá balanced cho thấy chênh accuracy/MF1 nhỏ, chưa đủ coi đây là nguyên nhân. N21 marginalF1 chỉ là phép thử giả thuyết, không đáp án mặc định.
- “Policy collapse/entropy collapse là nguyên nhân”: old completed runs train accuracy tăng, entropy giảm vừa phải (~16%), zero-variance groups khoảng25%; severe collapse/near-total saturation chưa được chứng minh.
- “Token saliency là faithful explanation”: sai mức kết luận. Magnitude không cho biết dấu semantics, và saliency hiện đo own allocation chứ không full swapped reward.
- “Shared-premise assignment chống bias hoàn toàn”: chỉ chứng minh bất biến row/class offsets của conditional allocation; hypothesis-specific bias vẫn tồn tại.
- “F1 utility là exact expected greedy test F1”: không đúng. Đây là conditional minibatch utility +sampled other-label vectors, với clipped/token-normalized policy surrogate.
- “Điểm tốt nhất qua nhiều inference protocols là kết quả chính”: không đúng. Format/backend/checkpoint selection phải được khóa trước test và report đúng scope.

## Lỗi kỹ thuật đã điều tra/sửa

| Vấn đề | Trạng thái / cách xử lý |
|---|---|
| Eval GPU OOM sau train | Tách train thành child, đợi exit trước inference; runconfig runtime memory hạ phù hợp. `empty_cache()` ở parent không đủ nếu process khác giữ GPU. |
| SFT target/chat prefix mismatch | Completion-only labels, non-thinking template chung, EOS giữ đúng; refuse overlength thay vì cắt target âm thầm. |
| Early SFT stopping proxy | Full-budget D3R9; final checkpoint chọn full-dev greedy. Không diễn giải teacher-forced metric là generation accuracy. |
| Dropout và detached score drift | Disable dropout cho direct replay; audit reference and score Jacobian. Active replay maxdiff0 ở passed smokes/runs. |
| Original/swap/saliency padding width khác | Có numerical BF16 confound. SourceR18 mới giữ conservative common global width; frozen old VMs không bị thay source tại chỗ. Không có measured effect size riêng nên không gọi đây là nguyên nhân toàn bộ gain/loss. |
| Sampled answer bị double-count trong marginalRL | Test RED thấy mask sai; sửa mask trước TRL KL/importance calculation. |
| Nghi unconditional advantage unsqueeze | Đã đọc installed TRL1.15: conditional dim1 check tồn tại; broadcast bug nghi ngờ **không được tái hiện**. |
| Setup/test stdoutR16/R17 không vào collector | Regression2fail → fix prefix →17 lightweight tests; source-only remote suite455passed sau bổ sung source teacher module cho test import. Không load teacher model. |
| Nghi R20 archive thiếu rollout | False alarm: `snapshot_files` loại nhưng `collect_logs` thêm main+smoke; exact main-run check xác nhận có. Broad suffix assertion trước đó match cả smoke. Không có omission fix cần làm. |
| CSV đọc đúng lúc writer rewrite | Partial observation/KeyError từng bị xem nhầm là training failure; đọc lại actual process/log. Không restart từ một lần poll lỗi. |
| R13 setup dependency race | Gate chạy trước install xong →21 import/dependency failures; giữ failed logs, chờ ready flag, rerun437passed rồi mới launch. |
| OAuth/network/account/quota | DNS từng lỗi, tokenrefresh bị Google buộc reauth, Drive quota đầy. Đây là infrastructure, không phải model failure. Không lặp lại mã OAuth cũ. |

## Regression/smoke evidence lịch sử

Đã có các full remote suite snapshots391/437/452/455/473/476tests ở các revision khác nhau. Những số này là **historical evidence**, không phải fresh test run của bản handoff hiện tại. Xem source revision/hash và ledger tương ứng. Handoff không chạy lại ML tests trên Mac.

Real GPU smoke yêu cầu finite nonzero policy/answerCE/direct gradients khi mode có channel đó, reference immutable, active dropout0, active score replay drift0. Một smoke pass chưa chứng minh held-out gain.

## Lỗi vận hành chưa khép lại

Checkpoint200 futility review của N21 chỉ là manual controller decision; observer không tự stop. Auth loss làm controller không xác minh/thực thi được. Session mới phải ghi đúng điều này, không nói gate đã bảo vệ compute. Các observer/monitor có timeout; timeout quan sát không xác nhận training terminal.
