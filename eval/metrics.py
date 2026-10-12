"""Accuracy, macro-F1, confusion matrix, accuracy theo chủ đề, tỉ lệ parse.
"""
from common.labels import LABELS


def compute_metrics(records: list[dict]) -> dict:
    n = len(records)
    correct = sum(1 for r in records if r["pred"] == r["gold"])
    confusion: dict[str, dict[str, int]] = {}
    for r in records:
        pred = r["pred"] if r["pred"] is not None else "unparsed"
        confusion.setdefault(r["gold"], {})
        confusion[r["gold"]][pred] = confusion[r["gold"]].get(pred, 0) + 1
    f1s = []
    for lab in LABELS:
        tp = sum(1 for r in records if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in records if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in records if r["gold"] == lab and r["pred"] != lab)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    topics: dict[str, list[int]] = {}
    for r in records:
        topics.setdefault(r["topic"], []).append(int(r["pred"] == r["gold"]))
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s),
        "parse_rate": sum(1 for r in records if r["parse_ok"]) / n if n else 0.0,
        "confusion": confusion,
        "acc_by_topic": {t: sum(v) / len(v) for t, v in sorted(topics.items())},
    }
