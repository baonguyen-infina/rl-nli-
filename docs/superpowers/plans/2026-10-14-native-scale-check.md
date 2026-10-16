# Teacher-free native RL scale check

The existing seed0 2B pilots remain below the .57–.58 dev Macro-F1 target. A fixed late-training audit of50 N4R5 rollouts found58% at the64-token cap; the incorrect subset contains role, quantity and unsupported-inference failures. These observations motivate a backbone-scale check, not a promise that greater capacity fixes reasoning. A5 already includes Qwen3.5-4B; it remains the student, with no teacher or distillation.

Cache official Qwen/Qwen3.5-4B weights only on Colab and record the commit. Preserve the running R7 sources and evaluations. On the first free G4, train D4R8 as the same gold-label SFT warm start, with shared base.yaml hyperparameters and checkpoint selection on dev. Register N4R8 for the existing allocation method at the shared400-step schedule, changing only model and same-seed warm start. Before a full RL run, G0N4R8 must pass one real GPU update, finite/nonzero loss-component gradients, frozen-reference and dropout audits. Launch the full method only if the pending2B methods still miss .57 or the scale check becomes the next needed experiment.

The Drive account has a finite15GiB limit despite the user's broad compute authorization. Preserve all existing archives. This follow-up backs up selected SFT/RL adapters, compact RL checkpoints100/200, and metadata; full optimizer checkpoints remain on Colab. Check projected archive sizes before starting. Never copy weights to the user's local computer. No test split. No training-seed claims from inference repeats and no cross-model causal claim for a performance gain. A successful4B method would still require matched4B controls before a paper claim about native RL.

- [ ] Cache the official student checkpoint and pin its recorded local revision.
- [ ] Register source/config snapshots without changing live R7 checkouts.
- [ ] Train a same-seed gold-only4B warm start after the current lane finishes.
- [ ] Reassess R7 outcomes, then run a4B GPU smoke and main method if warranted.

User requested stopping all work before office. All three Colab G4 assignments were terminated; local monitoring was stopped. R7 runs remain incomplete, and the queued4B SFT warm start never started. Previously verified completed checkpoints remain on Drive. Reinspect archived state and use fresh run IDs before restarting; do not infer an optimizer-resumable R7 checkpoint from partial progress logs.
