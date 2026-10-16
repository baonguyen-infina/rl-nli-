# R13 research boundary — internal, no paper results

The running cohort is unchanged. It tests finite-label conditional utility after sampled reasoning, with a matched accuracy-utility control. Its Macro-F1 utility samples the other examples' labels; it is not an exact expectation over the entire batch's label vector. Exact score-gradient replay refers only to the detached conditional utility currently supplied to each row.

Relevant primary work constrains novelty claims:

- [Likelihood-Based Reward Designs for General LLM Reasoning, 2026](https://arxiv.org/html/2602.03979v1) studies probability/log-probability reasoning rewards and supervised-gradient terms. Dense label likelihood plus RL is therefore not sufficient novelty. Its long-answer probability failures do not by themselves establish failure for a three-label normalized distribution.
- [Rao-Blackwellized Stochastic Gradients for Discrete Distributions, ICML 2019](https://proceedings.mlr.press/v97/liu19c.html) already studies variance reduction by summing discrete categories. Any conditional variance argument must cite this foundation and must not become an unsupported claim that the implemented clipped, token-normalized objective is an unbiased corpus-F1 gradient.
- [Maximum Expected F-Measure Training of Logistic Regression Models, HLT-EMNLP 2005](https://aclanthology.org/H05-1087/) is direct prior art on expected F-measure optimization. Metric alignment alone is not a new contribution.

A possible later diagnostic is to separate uncertainty from sampled reasoning and uncertainty from other examples' sampled class decisions. Do not launch an additional method merely because a new formula can be written. First obtain the locked cohort's actual greedy full-dev outcomes, compare its primary utility against the matched accuracy utility, and inspect which classes change. If both fail, additional generic reward variants need evidence before consuming another cohort.

The two interrupted warm-start evaluations had identical prediction SHA256s, establishing repeat agreement for that frozen checkpoint under this specific inference protocol. This does not establish repeatability for every future checkpoint or prove reasoning quality. The interrupted training produced no checkpoint100 evaluation, so its training accuracy cannot establish a method improvement.
