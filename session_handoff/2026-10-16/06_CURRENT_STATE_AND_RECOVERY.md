# Trạng thái cuối và quy trình tiếp nhận

**Điều chắc chắn mới nhất:** ngày16/10/2026, local Google credential hết hạn; refresh bị Google từ chối với thông báo yêu cầu reauthentication. Không còn khả năng xác minh live remote workload. Goal đã đặt blocked sau3turn cùng blocker. Ghi chép sau đây là last-known, không phải hiện trạng live.

## Các lane gần nhất

| Run | Colab session | Last verified |
|---|---|---|
| N20R20 | `rl-nli-resume-r18-g4` | Smoke pass; checkpoint100 full-dev verified và adapterarchive verified; SIGINT exact worker85155 tại observedstep109; ngay sau stop không train/eval và GPU0MiB. Parent pipeline/testworker còn cần xác minh terminal. |
| N21R20 | `rl-nli-r19-native-swap-2b-g4` | Smoke pass; checkpoint100 full-dev verified/archive verified; checkpoint200 observerPID81175 đã live. Train gần nhất khoảng158–166/400 trước monitoring gap. **Không biết hiện nay ở bước nào/đã xong chưa.** |
| N10R13 recovery | `rl-nli-r13-recovery-primary-g4` | Session name còn trong local CLI cache; không đủ biết assignment còn sống. Phải list server assignments. |

PID lịch sử N20 pipeline84073/test84072/train85155; N21 pipeline65868/test65867/train66953. **Không kill theo PID cũ**; phải kiểm tra `/proc/PID/cmdline`, runID và process start hiện tại.

Backup checkpoint100 đã ghi nhận, chưa query lại GitHub ở lần bàn giao:

| Run | Asset ID | Bytes | SHA256 |
|---|---:|---:|---|
| N20R20 | 644730748 | 123055736 | `e3ce358c8e42de181b79f0ac2c0221fb13ad7921181f65a5c8f373676c40b254` |
| N21R20 | 644678338 | 123264984 | `b00b55760f6e9624e6bfecf1b735b197861f585015c3b22c6dedaff79bcdb53d` |

Receipt và final archives N15/N16 tra trong `catalog/github_archive_index.json`. Các asset này nằm trong private draft release413873953; không download weights về Mac để restore.

## N20 deliberate stop

- `operators/native_r20_stop_futility.py` đã gửi SIGINT, có receipt remote `results/N20R20_Qwen3.5-2B_s0/early_stop_receipt.json`.
- Reason: checkpoint100 chưa compelling hơn lane nhanh hơn, unrestricted credit issue còn; đây là exploratory compute decision, **không preregistered stopping rule**.
- `paper_eligible=false`, `test_accessed=false` ở thời điểm stop, `do_not_retry_as_training_failure=true`.
- Mechanical cost/pipeline có thể ghi failed do training child bị SIGINT. Đây **không phải trigger restart**. Không setcostcomplete hoặc giả400complete.
- Phải đợi/kiểm tra testworker đã nhận costfailed và thoát trước khi đổi cost thànhaborted: worker cũ chỉ checkfailed, đổiaborted quá sớm có thể làm nó chờ8h.
- Giữ checkpoint100 adapter archive; pack một snapshot logs/metadata **mới sau khi mọi process kết thúc**, có CSV/rollouts/observer/dev/policy/stopreceipt/failurestdout. Không cần cứu adapter trong RAM tại109 cho discarded arm.
- Chỉ `colab stop --session rl-nli-resume-r18-g4` sau khi terminalstate và mới nhấtlogsbackup đã verify. Không dừng N21 cùng lệnh.

## N21 checkpoint200 compute rule

Policy đã ghi trước khi quan sát200: `evidence/internal/protocols/20261015/N21R20_checkpoint200_review.json`.

- `threshold = warm_full_dev_macro_f1 +0.005 =0.5724942572896857`.
- Nếu `max(cp100,cp200) < threshold`: stop exploratory run, preserve artifacts, **khôngtest**.
- Nếu pass: giữ original400budget, chọn max full-dev100/200/300/400, tieslater; once-only test theo manifest cũ.
- Neutral regressions chỉ descriptive, không thêm hiddencriterion.
- **Không có automatic stop daemon.** Observer200 chỉ evaluate và viết result; root controller phải quyết định. Authentication loss làm quyết định chưa được xác minh/thực thi.

Nếu train còn live sau200: đọc verifiedmetrics và áp dụng rule đã ghi. Nếu full400 hoặc test đã hoàn tất trong gap: ghi rõ deviation là gate chưa được thực thi, không retroactively coi một run đã hoàn tất là đã dừng theo rule; không đổi checkpoint theo test và không chạytestlại. Thu artifacts đã có rồi đánh giá tính hợp lệ của báo cáo đúng scope.

## Các file remote cần inspect read-only sau auth

```text
/content/N20R20_pipeline_status.json
/content/N20R20_test_status.json
/content/N20R20_checkpoint100_status.json
/content/N21R20_pipeline_status.json
/content/N21R20_test_status.json
/content/N21R20_checkpoint200_status.json
/content/rl-nli/results/<RUN>_Qwen3.5-2B_s0/cost.json
/content/rl-nli/results/<RUN>_Qwen3.5-2B_s0/ckpt_scores.json
/content/rl-nli/results/<RUN>_Qwen3.5-2B_s0/.test_claim/receipt.json
/content/rl-nli/results/<RUN>_Qwen3.5-2B_s0/test_verification.json
/content/rl-nli/results/<RUN>_Qwen3.5-2B_s0/github_backup/
```

Kiểm tra actual `ps` train/eval và `nvidia-smi`, không chỉ statusfile; đối chiếu source hashes với testregistration. Kernel exec CLI có thể return0 ngay cả khi Python traceback, nên phải parse output/status.

## Local operators

Đã dừng các local scripts bằng exact-command matching: monitorR20 PID3063, backupR20 PID16140, oldR13 logs20558, R18/R19 monitor48713, R13 monitor77862, finalizer88631. Đây là dừng **local monitoring**, không xác nhận GPU shutdown.

OAuth waiting exec handle52291 từng còn live; không chắc giữ được qua session mới. Tạo flow mới khi cần, không dùng authURL/code cũ. Credential file vẫn ở `~/.config/colab-cli/token.json`, không đọc/in ra token. SSH key và backup state ở `/private/tmp` có thể mất sau reboot. Không có credential copy trong handoff.

## Thứ tự tiếp tục

1. Đọc handoff, xác minh repo source/dirtytree, không reset.
2. `colab sessions` với tài khoản đúng; nếu reauth thì tạo1flow, nhận mã qua stdin của đúng process. Không yêu cầu accountmới chỉ vì authhết hạn.
3. Read-only list serverassignments, actualprocesses và artifacts; phân biệt lostobservation với workloadfailure.
4. Chốt N20abort+logsarchive; xử lý N21rule/deviation theo thực tế. Không allocate GPU mới trước bước này.
5. Verify privatearchive SHA/size/owner/draft, dữ liệu và frozen sources; archive trước releaseGPU.
6. Nếu tiếp tụcmonitor, chỉ monitorlanes còn cần; oldscripts mặc định2lanes có thể poll một aborted/releasedrun mãi. Chỉnh operatorcontroller riêng, không sửa frozen trainingcheckout.
7. Sau khi có evidence hiện tại, quyết định nghiên cứu tiếp theo bằng dev và controls; không đổi method vì thấy test thấp rồi tiếp tục tune trực tiếp trên test.
