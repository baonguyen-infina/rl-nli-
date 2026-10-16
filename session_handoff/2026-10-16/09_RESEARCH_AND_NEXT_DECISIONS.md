# Research, novelty và quyết định còn mở

Đây là tóm tắt review đã làm, không phải literature search mới hay kết luận mọi paper đã được đọc đầy đủ. Giữ các note gốc và danh mục source để session mới kiểm chứng nếu cần dùng trong paper.

## Prior work đã ràng buộc claim

- [GTA, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.56.pdf): Guess supervision, Think/Answer GRPO, masking và gradient conflict. User đã yêu cầu đọc paper này. Generic SFT+RL và supervised guess không phải novelty riêng của dự án.
- [Likelihood-Based Reward Designs, 2026](https://arxiv.org/html/2602.03979v1): likelihood reasoning rewards và supervised gradient terms đã có. Dense gold likelihood + RL tự thân chưa đủ.
- [Rao-Blackwellized Stochastic Gradients, ICML 2019](https://proceedings.mlr.press/v97/liu19c.html): discrete category marginalization/variance reduction có nền tảng cũ; trích đúng dù người dùng ưu tiên literature mới.
- [Maximum Expected F-Measure, HLT-EMNLP 2005](https://aclanthology.org/H05-1087/): expected F-measure optimization không mới. Không đặt novelty ở riêng “optimize F1”.
- [RLCR, 2025/revised 2026](https://arxiv.org/html/2507.16806v2): correctness + Brier confidence; NLI neutral là semantic label, không phải epistemic uncertainty.
- [TACO, July 2026 preprint](https://arxiv.org/html/2607.07976v1): harmful positive token credit/contextual tail risk là prior work đang phát triển. Magnitude saliency của mình không chứng minh semantic credit.
- [Neutralizing Bias using Entailment Graphs, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.705.pdf): counterfactual NLI augmentation/hypothesis-memory mitigation đã có; không giải thích tự động neutral regressions của mình.
- Review R16/R17 còn tra NLI-GRPO arXiv2504.18376, GRAIL arXiv2606.04889, In-Context RLVR 2026.findings-acl.1523 và ResRL PMLR306/lin26ak. Xem note gốc trước khi dùng substantive claim; không dựa vào tên paper để suy mechanism.

## Ý tưởng đã cân nhắc nhưng chưa có phép thử hợp lệ

- Hypothesis-only/reference calibration, inverse PoE hoặc train-only PoE: inverse baseline có thể thưởng full model sai khi baseline sai hơn; train-only PoE cần reliable H-only audit và có thể saturate. Đã defer, không được nói đã thử thành công.
- Confidence output/calibration: thay output protocol; chưa được thử trong registered R20. Không đồng nhất neutral với confidence thấp.
- Bounded correctness-gated trace reward + saliency: có thể isolate token credit trên nền bounded method, nhưng mới là candidate hypothesis, chưa registered/run/smoke. Không launch chỉ vì đã viết được formula.
- Tăng model size lên 4B: đã bị người dùng cấm. Teacher/distillation/rationalized data không hợp giới hạn.
- Multi-seed/multi-benchmark: cần về sau cho mạnh claim, người dùng hoãn để ưu tiên method. Không tự mở lúc chưa đủ gain.

## Tại sao chưa thể gọi A/A*

Test hiện có chưa cho meaningful, consistent RL gain so với SFT/policy-off. Novelty space đã có reward shaping, saliency, marginalization, NLI-RL. Native allocation có toán rõ, nhưng invariance không đồng nghĩa generalization/faithfulness. Một benchmark/một seed và adaptive method search sau opened test hạn chế confirmatory claim. Draft phải trung thực về các điểm này.

## Quy tắc chọn việc tiếp sau recovery

1. Thu và phân giải N20/N21 thực trạng trước, kiểm tra gate200 và archive. Không mở method thứ3 để né kết quả cũ.
2. Dùng dev paired predictions/per-class/errors chưa đổi để kiểm tra giả thuyết. Reward/entropy curve chỉ chẩn đoán optimizer, không thay metric.
3. Tách **policy reward**, **direct score gradient**, **answer CE**, **grouping**, **inference view** và **budget** khi đòi causal ablation. Nhiều biến cùng đổi chỉ là alternative candidate comparison.
4. Candidate tiếp phải giải được observed failure và có matched control cụ thể, stopping rule được thực thi bằng worker/controller đáng tin; không chỉ ghi policy file.
5. Lựa chọn trên dev, freeze trước test, report mọi kết quả đã registered. Test cũ đã opened thì ghi exploratory history, không tiếp tục loop test để đạt threshold rồi gọi unbiased.
6. Nếu gain chưa đủ thì nói thật. Không dùng resources/vòng lặp/văn phong paper để bù thiếu empirical evidence.
