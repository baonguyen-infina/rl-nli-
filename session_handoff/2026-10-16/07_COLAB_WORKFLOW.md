# Workflow triển khai, chạy và lưu bằng chứng

## CLI và môi trường

CLI hiện có trên Mac:

```text
/Users/infina/.local/share/uv/tools/google-colab-cli/bin/colab
/Users/infina/.local/share/uv/tools/google-colab-cli/bin/python
```

CLI không nằm trong PATH ở một số exec shells. Dùng absolute path hoặc thêm PATH trong process riêng; không repurpose HOME/CODEX_HOME. Python hệ thống dùng `python3`, không giả định có `python`.

Các lệnh đúng đã kiểm tra: `colab sessions` (không phải `sessions list`), `colab stop --session NAME`. `colab exec --session NAME --file SCRIPT --timeout45 --env KEY=VALUE` chạy source script trên Colab. Tra `--help` của install hiện tại trước lệnh tạo GPU; không suy ra tier từ VRAM. Historical G4 được server xác nhận và GPU thực tế là RTX PRO6000 Blackwell Server Edition khoảng96GiB; H200 thuộc paper trước, không tự ghi H200 cho RLcohort.

Các môi trường thành công gần đây dùng TRL1.15.0, Transformers5.17.0, vLLM0.31.0 và BF16 LoRA. `requirements.txt` vẫn chưa pin toàn bộ dependencies; muốn reproduce dùng environment/source snapshots của run, không `pip install -U` tùy tiện lên revision mới rồi coi tương đương.

## Một run mới hợp lệ

1. **Review trước compute**: chọn một hypothesis từ dev diagnostics, quyết định đối chứng, data/exposure/budget/evaluator; ưu tiên reuse baseline đã có. Giới hạn tối đa3 G4, không tự lấp slot trống.
2. **Register**: thêm run ID và smoke ID trong matrix, profile chung base; generate YAML. Giữ source cũ của completed/live runs. Run ID mới nếu method/horizon/evaluator đổi.
3. **Deploy source-only**: tạo bundle source/config/tests nhỏ; upload tới Colab. Restore exact warm adapter **server-side** từ archive có checksum. Không lấy weights về Mac.
4. **Environment/data gates**: dependencies-ready flag, pinned revision, train/dev hash, đủ GPU/disk, no UID features.
5. **Remote regression**: RED/GREEN nếu đổi code, rồi full suite trên Colab/source-only checkout. Gate phải chặn launch khi install chưa xong.
6. **Real GPU one-update smoke**: finite nonzero expected gradient channels, frozen identical reference, dropout0, active replay drift0, masks/padding/exposure đúng. Không suy benchmark từ smoke.
7. **Main**: mọi training qua `python -m scripts.run --run ID --seed 0 --phase all`; ghi frozen config, source hashes và initial full-dev. Training ở child để allocations giải phóng trước final evaluation.
8. **Monitoring**: actual worker process, last step/cost/source hash, GPU utilization; poll60–120s hoặc lâu hơn khi ổn định. Không timeout một poll là restart. Không chạy nhiều duplicate monitors cho cùng lane.
9. **Checkpoint readout**: save100/200/300/400, full-dev greedy; early observers có thể run nếu free VRAM>20GiB và frozen hash verified. Selection reuse các metrics này, không chạy inference repeats rồi chọn score cao.
10. **Selection/test**: cost complete + dev selection validated + manifest locked; once-only test bằng runner. Report result dù cao/thấp, không dùng test đổi config.
11. **Backup/release**: logs/checkpoints/selected adapter/source/test artifacts archive checksum; private owner/draft verified; chỉ trả VM khi terminal/no owned worker và off-VM backup verified.

## Core commands — chạy ở Colab, không chạy ML trên Mac

```bash
python -m configs.matrix
python -m pytest -q
python -m scripts.run --run G0N21R20P --seed 0 --phase all
python -m scripts.run --run N21R20 --seed 0 --phase all
```

Đây là minh họa entrypoint của các run lịch sử, **không phải chỉ thị replay chúng**. Main ID đã tồn tại sẽ refuse overwrite; không xóa output để ép chạy lại. Đừng launch smoke/main lịch sử mà chưa đọc recovery decision.

G4 test đi qua `scripts.run --phase test --protocol LOCKED_MANIFEST`, không gọi test bằng predict CLI trần. `eval/final_protocol.py` xác minh frozen artifact hash trước truy cập test; `.test_claim` receipt bảo vệ once-only execution.

## Backup hiện tại

- Repository private owned: `baonguyen-infina/rl-nli-experiment-archives`; draft release ID **413873953**. Chưa publish và không được tự publish.
- `native_r20_checkpoint_backup_pack.py` pack **trên Colab**. Event checkpoint100/200/300/400 giữ compact adapter+logs; `logs-<timestamp>` giữ metadata/logs/source không weights; final cần test verified/cost complete.
- `native_github_archive_relay.py` stream Colab→bounded RAM→GitHub với buffer≤1MiB; không ghi model archive ra Mac disk. Cả source SHA và GitHub asset digest/size phải khớp. Token GitHub chỉ local, Colab token chỉ gửi đúng host Colab.
- Receipt local ở `artifacts/analysis/github_backup_receipts/`, remote copy trong output `github_backup/`. Bản bàn giao giữ receipts/index trong evidence/catalog, không giữ asset weights.
- Periodic logs900s, checkpoints khi fully saved, failure logs khi failure phase đổi. Final test archiver riêng với periodic checkpoint backup, tránh hai final asset cùng tên, nội dung khác.
- `snapshot_files` và `collect_logs` cùng đóng góp. Dùng exact main run path kiểm tra rollout, vì smoke có cùng basename. Giữ cả stdout setup/test/failure prefixes.
- SSH key `/private/tmp/native_r13_ssh_key` và state `/private/tmp/native_github_archive_state.json` là local private; không copy vào handoff hay print. Nếu mất/reboot, thiết lập capability/state mới từ account đúng, không đoán token.

## Những việc chưa nên tự động

Không sửa frozen training source trên VM đang chạy. Không đặt cost complete để unblock scripts. Không chạy test nếu abort. Không xem một `test_verified_backup_pending` là archive đã xong. Không release VM vì pipeline parent exit mà evaluator/archiver còn làm. Không kéo entire archive weights về Mac chỉ để đọc metadata; dùng server-side extract hoặc range/stream có giới hạn.

Operators trong `operators/` là bản chụp phục hồi; một số import module từ `/private/tmp`, hardcode2 lanes/PID/deadline. Chép lại đúng helpers, đọc source rồi mới execute. Chúng không phải self-contained cloud scheduler và không có guarantee tiếp tục monitor khi laptop offline.
