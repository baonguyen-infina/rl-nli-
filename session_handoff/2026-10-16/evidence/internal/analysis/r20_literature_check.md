# R20 focused literature check

Checked against primary sources on 2026-10-16. These are method/novelty notes, not paper experimental results. No new training job is justified solely by these papers.

- [GTA, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.56.pdf) supervises the Guess segment and applies GRPO to Think/Answer, with masking and gradient-conflict handling. Combining label supervision and reasoning RL is therefore an existing foundation, not sufficient novelty for this project.
- [Beyond Binary Rewards / RLCR](https://arxiv.org/html/2507.16806v2) was first submitted in July2025 and revised in May2026. It augments correctness rewards with a Brier score for generated numerical confidence. It supports checking calibration after correctness-only RL, but does not establish that a semantic NLI neutral label should mean low confidence. Introducing confidence output would change the present protocol and is not authorized as an in-place live-run edit.
- [TACO, July2026 preprint](https://arxiv.org/html/2607.07976v1) studies harmful positive credit on contextually implausible low-probability tokens and attenuates their positive updates. Token-level credit calibration is already active prior work. Our absolute gradient-magnitude saliency is different and does not, by itself, prove semantic faithfulness or identify whether a token helped the correct answer.
- [Neutralizing Bias in LLM Reasoning using Entailment Graphs, Findings ACL2025](https://aclanthology.org/2025.findings-acl.705.pdf) creates counterfactual inference training examples from extracted predicate relations and typed argument substitutions. Counterfactual NLI augmentation and reducing hypothesis-memory shortcuts are therefore existing lines. This paper does not demonstrate the cause of our observed neutral regressions.

## Decision carried into current experiments

Keep N20/N21 frozen and inspect checkpoint100 dev predictions under the same greedy-label view. Examine repairs versus regressions for neutral and contradiction as well as total performance. Calibration or shortcut explanations remain hypotheses until measured. No third G4, teacher model, distillation, confidence-output change, or reward change is introduced during these registered runs.

The paired dev-only diagnostic is in `r20_paired_dev_diagnostic.json`. Its checkpoint-selection effects preclude treating the descriptive paired p-values as confirmatory tests. Both present methods also include supervised/direct-score gradient terms; an eventual claim of an RL-specific contribution requires a matched policy-off control, rather than attributing all improvement to the policy objective.
