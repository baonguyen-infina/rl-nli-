"""Persist rollout evidence using the same parser and record schema as evaluation."""
import json
from pathlib import Path

from common.io import write_train_log
from data.prepare import get_rows
from eval.records import build_record


def with_rollout_audit(trainer_cls, cfg, out_dir):
    metadata = {(r['premise'], r['hypothesis'], r['label']): r for r in get_rows('train')}
    path = Path(out_dir) / 'rollouts_train.jsonl'

    class AuditedTrainer(trainer_cls):
        audit_batch = 0

        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            with path.open('a', encoding='utf-8') as f:
                for index, (row, ids, mask) in enumerate(zip(
                        inputs, result['completion_ids'], result['completion_mask'])):
                    tokens = ids[mask.bool()].detach().cpu().tolist()
                    raw = self.processing_class.decode(tokens, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)
                    source = metadata[(row['premise'], row['hypothesis'], row['gold'])]
                    record = build_record(source, raw, cfg['format'], cfg.get('guess', False), len(tokens))
                    record.update(step=self.state.global_step, audit_batch=self.audit_batch,
                                  advantage=float(result['advantages'][index].detach().cpu()))
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.audit_batch += 1
            return result

    return AuditedTrainer


def train_log_callback(out_dir):
    from transformers import TrainerCallback

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            write_train_log(state.log_history, Path(out_dir) / 'train_log.csv')

    return LogCallback()
