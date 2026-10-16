"""Gold-label guess supervision and RL reasoning; no teacher rationales.

flat_fallback is a separately ablated variant that retains supervision when
the RL gradient is zero. The vanilla GTA cosine gate is also available.
"""


def guess_content_mask(text, offsets):
    start = text.find("<guess>")
    if start < 0:
        return [1] * len(offsets)
    start += len("<guess>")
    end = text.find("</guess>", start)
    if end < 0:
        end = len(text)
    return [0 if b > start and a < end else 1 for a, b in offsets]


def use_supervised_loss(dot, rl_norm_squared, sft_norm_squared,
                        cos_constraint=True, flat_fallback=False):
    if not cos_constraint:
        return True
    if rl_norm_squared == 0:
        return flat_fallback and sft_norm_squared > 0
    return dot > 0


def make_gta_trainer(cfg):
    from train.common import chat_template_ids
    from common.labels import LABELS
    from formats.parser import parse
    import torch
    import torch.nn.functional as F
    from trl import GRPOTrainer

    class GTATrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            tok = self.processing_class
            device = result["completion_ids"].device
            ids_list, labels_list, masks, rule_positions, supports = [], [], [], [], []
            if cfg.get('rule_supervision'):
                encoded = [tok.encode(' ' + status, add_special_tokens=False)
                           for status in ['match', 'missing', 'conflict']]
                if any(len(ids) != 1 for ids in encoded):
                    raise ValueError('Rule supervision requires one-token space-prefixed status words')
                self.rule_status_ids = [ids[0] for ids in encoded]
            for row, ids, active in zip(inputs, result["completion_ids"], result["completion_mask"]):
                prefix = chat_template_ids(tok, row["prompt"]) + tok.encode(
                    "<guess>", add_special_tokens=False)
                gold = tok.encode(row["gold"], add_special_tokens=False)
                ids_list.append(prefix + gold)
                labels_list.append([-100] * len(prefix) + gold)

                tokens = ids[:int(active.sum())].tolist()
                raw = tok.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if cfg.get('rule_supervision'):
                    from train.rule_supervision import alignment_status_positions
                    parsed = parse(tok.decode(tokens, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False), 'align', True)
                    positions = alignment_status_positions(tokens, tok, self.rule_status_ids) if parsed.format_ok else []
                    if len(positions) != len(parsed.align_lines or []):
                        positions = []
                    rule_positions.append(positions + [-1] * (6 - len(positions)))
                    if cfg.get('rule_support') == 'schema':
                        from train.rule_supervision import schema_status_support
                        allowed = schema_status_support(parsed.align_lines or []) if positions else []
                        supports.append(allowed + [[True, True, True]] * (6 - len(allowed)))
                end = raw.find("</guess>")
                end = len(raw) if end < 0 else end
                offsets, previous = [], 0
                if "<guess>" in raw:
                    for j in range(len(tokens)):
                        if previous >= end:
                            break
                        current = len(tok.decode(tokens[:j + 1], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False))
                        offsets.append((previous, current))
                        previous = current
                mask = guess_content_mask(raw, offsets)
                masks.append(mask + [1] * (ids.numel() - len(mask)))

            length = max(map(len, ids_list))
            result["guess_input_ids"] = torch.tensor(
                [[tok.pad_token_id] * (length - len(x)) + x for x in ids_list], device=device)
            result["guess_labels"] = torch.tensor(
                [[-100] * (length - len(x)) + x for x in labels_list], device=device)
            result["guess_attention_mask"] = torch.tensor(
                [[0] * (length - len(x)) + [1] * len(x) for x in ids_list], device=device)
            mask = torch.tensor(masks, device=device, dtype=result["completion_mask"].dtype)
            result["tool_mask"] = mask * result.get("tool_mask", torch.ones_like(mask))
            if cfg.get('rule_supervision'):
                result['rule_positions'] = torch.tensor(rule_positions, device=device)
                result['rule_gold'] = torch.tensor([LABELS.index(row['gold']) for row in inputs], device=device)
                if cfg.get('rule_support') == 'schema':
                    result['rule_support'] = torch.tensor(supports, device=device, dtype=torch.bool)
            return result

        def _compute_loss(self, model, inputs):
            rl_loss = super()._compute_loss(model, inputs)
            if not model.training:
                return rl_loss
            labels = inputs["guess_labels"]
            keep = int((labels != -100).sum(dim=1).max()) + 1
            output = model(input_ids=inputs["guess_input_ids"],
                           attention_mask=inputs["guess_attention_mask"],
                           use_cache=False, logits_to_keep=keep)
            logits = output.logits[:, :-1].contiguous()
            targets = labels[:, -logits.size(1):].contiguous()
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                 targets.view(-1), ignore_index=-100)
            sft_loss = ce / self.current_gradient_accumulation_steps
            params = [p for p in model.parameters() if p.requires_grad]
            rgrads = torch.autograd.grad(rl_loss, params, retain_graph=True, allow_unused=True)
            sgrads = torch.autograd.grad(sft_loss, params, retain_graph=True, allow_unused=True)
            zero = torch.zeros((), device=rl_loss.device)
            rn = sum((g.float().square().sum() for g in rgrads if g is not None), zero)
            sn = sum((g.float().square().sum() for g in sgrads if g is not None), zero)
            dot = sum(((r.float() * s.float()).sum() for r, s in zip(rgrads, sgrads)
                       if r is not None and s is not None), zero)
            include = use_supervised_loss(dot.item(), rn.item(), sn.item(),
                                           cfg.get("cos_constraint", True),
                                           cfg.get("flat_fallback", False))
            cosine = (dot / (rn.sqrt() * sn.sqrt()).clamp(min=1e-20)).item()
            for key, value in (("gta/guess_ce", ce.item()), ("gta/gradient_cosine", cosine),
                               ("gta/used_supervised", float(include)),
                               ("gta/flat_rl", float(rn.item() == 0))):
                self._metrics["train"][key].append(value)
            total = rl_loss + cfg["gta"]["sft_weight"] * sft_loss if include else rl_loss
            if cfg.get('rule_supervision'):
                from train.rule_supervision import rule_supervision_loss
                rule_ce, fraction = rule_supervision_loss(model, inputs, self.rule_status_ids)
                self._metrics['train']['gta/rule_example_fraction'].append(fraction)
                self._metrics['train']['gta/rule_ce'].append(0.0 if rule_ce is None else rule_ce.item())
                if rule_ce is not None:
                    total = total + cfg['gta']['rule_weight'] * rule_ce / self.current_gradient_accumulation_steps
            return total

    return GTATrainer
