# Kết quả test và manuscript

Nguồn số có thể kiểm tra máy được: `catalog/test_results.json` và CSV tương ứng; bản tổng hợp cũ `evidence/internal/analysis/final_test_runs.csv`;14 local test result folders có config/cost/once-only receipt/predictions. Hai follow-upN15/N16 được giữ qua remote verification metadata +private GitHub final archive receipts.

## Các điểm chính đã xác minh

| Run / view | Test Macro-F1 | Diễn giải |
|---|---:|---|
| HD3R9L | 0.5637477952 | SFT label-only baseline |
| HN1R9L | 0.5636748345 | OrdinaryRL label-only |
| HN4R9L | 0.5610993774 | Direct native allocation setup |
| HN5R10L | 0.5560625736 | Joint allocation +policy shaping |
| HN6R11L | 0.5559095727 | Correctness-gated joint |
| HN7R12P200L | 0.5637640620 | Random grouping, checkpoint200 view |
| HN0R11C | 0.5681789181 | **Policy-off** allocation control, CoT view; không được gọi là RL đạt điểm này |
| N15R18 | 0.5651534564 | Bounded swap/random, full400 selected on dev |
| N16R19 | 0.5606293811 | Bounded swap/native, matched400 |

Tất cả rows trên seed0/n1000/parse1.0. Bảng đầy đủ còn các CoT/label views khác, không chỉ những row được trích ở đây. Không có test≥.58 được xác minh. Việc một số row vượt.55 không hoàn thành mục tiêu nâng lên.58. Không so chênh.000016 giữa SFT và một RL view như bằng chứng mạnh.

N15/N16 thuộc exploratory follow-up sau cohort test cũ đã mở. Candidate/checkpoint rule được khóa trước test của từng follow-up, và không retune theo test đó; nhưng không được mô tả toàn bộ quá trình như một single untouched confirmatory test study.

N20: deliberate abort; không có test được phép từ interrupted run. N21: **chưa biết** testworker đã chạy hay chưa sau mất auth. File cũ không thay trạng thái thực tế. Phải inspect `.test_claim`, `test_verification.json` và `cost_test.json` rồi mới nói có kết quả mới.

## Paper hiện có

- Source `paper/main.tex`, bib `paper/refs.bib`, ACL style ở `paper/acl.sty`/`acl_natbib.bst`; bản copy nằm `snapshot/repo/paper/`.
- Working title quanh conditional native label allocation teacher-free SFT/RL. Đây không phải bản paper hybrid người dùng đã đưa trước đó.
- Build/report cũ từ14 test views;7pages; SHA256 PDF `f6355e6bb9c555e8ee79bb118442396803ff7d6ae51a66e6f6ec5cff1b14f130` theo existing provenance/visual QA. Bản handoff có đối chiếu lại checksum file; không render/build lại.
- Existing report nói build returncode0 nhưng log có BibTeX warnings/errors-ignored và underfull boxes. Visual QA lịch sử nói7pages passed/no overfull. Không đổi thành tuyên bố fresh warning-free compile.
- `analysis/collect.py --split test` chỉ nhận costcomplete+receiptcompleted và kiểm tra integrity outputs. `analysis/paper_report.py`, `make_tables.py`, `make_figs.py`, `stats.py` sinh artifact paper; không chép số tay.
- N15/N16 không tự được thêm vào14-view tables cũ. Cần cohort/protocol update có provenance phù hợp, không sửa abstract bằng số dev.
- Single-seed premise-cluster bootstrap/paired intervals đo sampling uncertainty có điều kiện trên fitted models, không đo training-seed variance.

## Chưa đủ để tuyên bố

Hiện chưa chứng minh lợi ích RL-specific, superiority ổn định hơn matched controls, generalization nhiều datasets hoặc causal faithfulness. User đã hoãn multi-seed nên không tự đốt compute multi-seed khi chưa giải được method. Tối thiểu phải có meaningful effect +appropriate ablations/compute-matching và kiểm chứng mechanism trước khi nâng claim. Đạt threshold đơn lẻ không tự xác định hạng hội nghị.

Paper hybrid người dùng cung cấp báo XLM-R+Qwen2.5-3B self-attention ViANLI MF1 55.72±.38 và average74.72 ba datasets. Đây là nội dung người dùng đưa, không được tái dùng như result của code RL này; model/protocol/seeds khác. Không có URL publication của paper hybrid được kiểm chứng trong bộ bàn giao.
