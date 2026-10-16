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
    
    precisions, recalls, f1s, supports = {}, {}, {}, {}
    for lab in LABELS:
        tp = sum(1 for r in records if r["gold"] == lab and r["pred"] == lab)
        fp = sum(1 for r in records if r["gold"] != lab and r["pred"] == lab)
        fn = sum(1 for r in records if r["gold"] == lab and r["pred"] != lab)
        supp = tp + fn
        supports[lab] = supp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        precisions[lab] = p
        recalls[lab] = r
        f1s[lab] = f1

    macro_p = sum(precisions.values()) / len(LABELS)
    macro_r = sum(recalls.values()) / len(LABELS)
    macro_f1 = sum(f1s.values()) / len(LABELS)
    acc = correct / n if n else 0.0
    parse_rate = sum(1 for r in records if r["parse_ok"]) / n if n else 0.0

    report = (
        f"\n{'='*70}\n"
        f"{'CLASSIFICATION REPORT':^70}\n"
        f"{'='*70}\n"
        f"{'Class':<18} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>12}\n"
        f"{'-'*70}\n"
    )
    for lab in LABELS:
        report += f"{lab:<18} {precisions[lab]:>10.4f} {recalls[lab]:>10.4f} {f1s[lab]:>10.4f} {supports[lab]:>12d}\n"
    report += (
        f"{'-'*70}\n"
        f"{'Macro Avg':<18} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {n:>12d}\n"
        f"{'Accuracy':<18} {'':>10} {'':>10} {acc:>10.4f} {n:>12d}\n"
        f"{'Parse Rate':<18} {'':>10} {'':>10} {parse_rate:>10.4f} {n:>12d}\n"
        f"{'='*70}\n"
    )

    topics: dict[str, list[int]] = {}
    for r in records:
        topics.setdefault(r["topic"], []).append(int(r["pred"] == r["gold"]))
    
    per_class = {
        lab: {
            "precision": round(precisions[lab], 4),
            "recall": round(recalls[lab], 4),
            "f1": round(f1s[lab], 4),
            "support": supports[lab],
        }
        for lab in LABELS
    }

    return {
        "n": n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "parse_rate": parse_rate,
        "per_class": per_class,
        "report": report,
        "confusion": confusion,
        "acc_by_topic": {t: sum(v) / len(v) for t, v in sorted(topics.items())},
    }
