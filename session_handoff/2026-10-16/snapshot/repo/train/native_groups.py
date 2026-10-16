"""Bounded native contrasts and conditional label-allocation utilities.

Grouping metadata belongs to the training loss, never to prompt text. Allocation
conditions on an observed label multiset; it measures relative assignment and
does not establish correct absolute answers or semantic evidence grounding.
"""

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, permutations
import math
import operator
import random
import unicodedata


LABELS = ("entailment", "neutral", "contradiction")


def _premise(row):
    value = row["premise"]
    if not isinstance(value, str):
        raise ValueError("premise must be a string")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _bound(max_size):
    if not isinstance(max_size, int) or not 1 <= max_size <= 3:
        raise ValueError("max_size must be an integer from 1 to 3")


def build_native_blocks(rows, seed=0, max_size=3):
    """Partition all original rows by normalized premise, without using labels."""
    _bound(max_size)
    groups = defaultdict(list)
    for row in rows:
        groups[_premise(row)].append(row)
    rng = random.Random(seed)
    blocks = []
    for members in groups.values():
        members = members.copy()
        rng.shuffle(members)
        blocks.extend(members[i:i + max_size] for i in range(0, len(members), max_size))
    return blocks


def _validate_blocks(blocks, max_size=3):
    _bound(max_size)
    if any(not block or len(block) > max_size for block in blocks):
        raise ValueError("blocks must contain between 1 and max_size rows")


def build_random_blocks(native_blocks, seed=0):
    """Match label slots and exact exposure while separating premise texts.

    Rows are permuted inside label buckets, then repaired with same-label swaps.
    The bounded construction returns a complete control or raises ValueError;
    it never silently retains a shared-premise tail. Exhausting repair is not
    a mathematical proof that a feasible assignment does not exist.
    """
    _validate_blocks(native_blocks)
    if not native_blocks:
        return []
    slots = defaultdict(list)
    buckets = defaultdict(list)
    premise_by_object = {}
    for group, block in enumerate(native_blocks):
        for slot, row in enumerate(block):
            slots[row["label"]].append((group, slot))
            buckets[row["label"]].append(row)
            premise_by_object[id(row)] = _premise(row)

    premise_counts = Counter(premise_by_object[id(row)] for block in native_blocks for row in block)
    if any(count > len(native_blocks) for count in premise_counts.values()):
        raise ValueError("infeasible different-premise control: too many rows share a premise")
    available = {label: {premise_by_object[id(row)] for row in rows} for label, rows in buckets.items()}
    # Necessary distinct-premise conditions for each observed slot template.
    templates = {tuple(sorted(row["label"] for row in block)) for block in native_blocks}
    for labels in templates:
        for size in range(1, len(labels) + 1):
            for subset in combinations(labels, size):
                if len(set().union(*(available[label] for label in subset))) < size:
                    raise ValueError("infeasible different-premise control for a label template")

    rng = random.Random(seed)
    for _ in range(128):
        assigned = [[None] * len(block) for block in native_blocks]
        for label, positions in slots.items():
            members = buckets[label].copy()
            rng.shuffle(members)
            for (group, slot), row in zip(positions, members):
                assigned[group][slot] = row
        failed = False
        for group, block in enumerate(assigned):
            for slot, row in enumerate(block):
                original_premise = premise_by_object[id(row)]
                other_premises = {premise_by_object[id(item)] for j, item in enumerate(block) if j != slot}
                if original_premise not in other_premises:
                    continue
                candidates = slots[row["label"]]
                start = rng.randrange(len(candidates))
                repaired = False
                for offset in range(len(candidates)):
                    donor_group, donor_slot = candidates[(start + offset) % len(candidates)]
                    if donor_group == group:
                        continue
                    donor = assigned[donor_group][donor_slot]
                    if premise_by_object[id(donor)] in other_premises:
                        continue
                    donor_others = {premise_by_object[id(item)] for j, item in enumerate(assigned[donor_group]) if j != donor_slot}
                    if original_premise in donor_others:
                        continue
                    assigned[group][slot], assigned[donor_group][donor_slot] = donor, row
                    repaired = True
                    break
                if not repaired:
                    failed = True
                    break
            if failed:
                break
        if not failed:
            return assigned
    raise ValueError("different-premise control is infeasible or exhausted bounded repair; no partial control returned")


def pad_blocks(blocks, max_size=3):
    """Copy rows to a flat rectangular layout; duplicate padding has weight zero."""
    _validate_blocks(blocks, max_size)
    padded = []
    for group, block in enumerate(blocks):
        for slot in range(max_size):
            item = dict(block[slot] if slot < len(block) else block[0])
            item.update(group_key=group, slot=slot, slot_weight=int(slot < len(block)))
            padded.append(item)
    return padded


def build_scheduled_random_blocks(native_blocks, group_order, groups_per_batch=8, seed=0):
    """Match random controls inside each scheduled update's exact row cohort.

    A bounded pilot covers a partial epoch: globally permuting examples would
    change the observed training exposure. Unvisited groups retain their rows.
    """
    if groups_per_batch < 1 or len(set(group_order)) != len(group_order):
        raise ValueError('Invalid scheduled group order')
    if any(not 0 <= g < len(native_blocks) for g in group_order):
        raise ValueError('Group index outside native blocks')
    result = [list(block) for block in native_blocks]
    complete = len(group_order) // groups_per_batch * groups_per_batch
    for start in range(0, complete, groups_per_batch):
        indices = group_order[start:start+groups_per_batch]
        matched = build_random_blocks([native_blocks[g] for g in indices], seed=seed + start)
        for g, block in zip(indices, matched, strict=True):
            result[g] = block
    return result


def _gold_index(value):
    if isinstance(value, str):
        try:
            return LABELS.index(value)
        except ValueError as exc:
            raise ValueError("unknown active gold label") from exc
    try:
        index = operator.index(value)
    except TypeError as exc:
        raise ValueError("active gold must be a label or integer class index") from exc
    if not 0 <= index < 3:
        raise ValueError("active gold class index must be 0, 1 or 2")
    return index


@lru_cache(maxsize=64)
def _assignments(gold):
    return tuple(sorted(set(permutations(gold))))


def log_allocation(scores, gold, active_mask):
    """Return scalar logQ for at most three slots and three canonical classes.

    Scores may be logits or log probabilities: per-slot normalizers cancel.
    Inactive slots, including their gold labels and nonfinite values, are ignored.
    """
    if not (len(scores) == len(gold) == len(active_mask)) or len(scores) > 3:
        raise ValueError("scores, gold and active_mask must have matching bounded slot lengths")
    if any(len(slot) != 3 for slot in scores):
        raise ValueError("each score slot must have exactly three classes")
    if any(weight not in (0, 1) for weight in active_mask):
        raise ValueError("active_mask must contain only zero or one")
    active = [i for i, weight in enumerate(active_mask) if weight]
    labels = tuple(_gold_index(gold[i]) for i in active)
    selected = [[float(value) for value in scores[i]] for i in active]
    if any(not math.isfinite(value) for slot in selected for value in slot):
        raise ValueError("active scores must be finite")
    if len(set(labels)) <= 1:
        return 0.0
    target = math.fsum(slot[label] for slot, label in zip(selected, labels))
    candidates = [math.fsum(slot[label] for slot, label in zip(selected, assignment)) for assignment in _assignments(labels)]
    peak = max(candidates)
    normalizer = peak + math.log(math.fsum(math.exp(value - peak) for value in candidates))
    return min(0.0, target - normalizer)


def tensor_log_allocation(class_scores, gold, weights):
    """Differentiable logQ[group, rollout] from scores[group, rollout, slot, 3].

    gold and binary weights have shape [group, slot]. Same-label and empty
    contrasts return a graph-connected zero; every padding gradient is zero.
    Half-precision inputs are promoted for the small logsumexp calculation.
    """
    import torch

    if class_scores.ndim != 4 or class_scores.shape[-1] != 3:
        raise ValueError("class_scores must have shape [group, rollout, slot, 3]")
    groups, rollouts, slots, _ = class_scores.shape
    if not 1 <= slots <= 3 or rollouts < 1:
        raise ValueError("allocation requires 1..3 slots and at least one rollout")
    if tuple(gold.shape) != (groups, slots) or tuple(weights.shape) != (groups, slots):
        raise ValueError("gold and weights must have shape [group, slot]")
    if not class_scores.is_floating_point() or gold.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("scores must be floating point and gold must contain integer indices")
    mask_rows = weights.detach().cpu().tolist()
    if any(weight not in (0, 1) for row in mask_rows for weight in row):
        raise ValueError("weights must contain only zero or one")
    gold_rows = gold.detach().cpu().tolist()
    working = class_scores.float() if class_scores.dtype in (torch.float16, torch.bfloat16) else class_scores
    if groups == 0:
        return working.sum(dim=(-2, -1)) * 0.0
    outputs = []
    for group in range(groups):
        active = [i for i, weight in enumerate(mask_rows[group]) if weight]
        labels = tuple(_gold_index(gold_rows[group][i]) for i in active)
        selected = working[group][:, active, :]
        if not torch.isfinite(selected).all().item():
            raise ValueError("active scores must be finite")
        if len(set(labels)) <= 1:
            outputs.append(torch.nan_to_num(working[group, :, 0, 0]) * 0.0)
            continue
        index = torch.tensor(labels, dtype=torch.long, device=working.device).expand(rollouts, -1).unsqueeze(-1)
        target = selected.gather(-1, index).sum(dim=(-2, -1))
        candidates = []
        for assignment in _assignments(labels):
            index = torch.tensor(assignment, dtype=torch.long, device=working.device).expand(rollouts, -1).unsqueeze(-1)
            candidates.append(selected.gather(-1, index).sum(dim=(-2, -1)))
        outputs.append(target - torch.logsumexp(torch.stack(candidates), dim=0))
    return torch.stack(outputs)
