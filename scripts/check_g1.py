"""Kiểm cổng G1: E3 > zero-shot (E0–E2) trên dev và parse E3 >= 99%; đủ E0–E3, B1, hyp probe."""
import sys

from common.io import ARTIFACTS, RESULTS, read_json

REQUIRED = ["E0", "E1", "E2", "E3", "B1"]


def evaluate_gate(metrics: dict) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for rid in REQUIRED:
        if rid not in metrics:
            lines.append(f"thiếu kết quả {rid}")
            ok = False
    if not ok:
        return False, lines
    zs = max(metrics[r]["accuracy"] for r in ("E0", "E1", "E2"))
    e3 = metrics["E3"]
    better = e3["accuracy"] > zs
    parse_ok = e3["parse_rate"] >= 0.99
    lines.append(f"E3 acc {e3['accuracy']:.4f} vs best ZS {zs:.4f}: {'OK' if better else 'FAIL'}")
    lines.append(f"E3 parse {e3['parse_rate']:.4f} >= 0.99: {'OK' if parse_ok else 'FAIL'}")
    lines.append(f"B1 acc {metrics['B1']['accuracy']:.4f} (tham chiếu)")
    return better and parse_ok, lines


def main() -> None:
    metrics = {}
    for rid in REQUIRED:
        for d in sorted(RESULTS.glob(f"{rid}_*_s0")):
            f = d / "metrics_dev.json"
            if f.exists():
                metrics[rid] = read_json(f)
    ok, lines = evaluate_gate(metrics)
    probe = (ARTIFACTS / "hyp_probe" / "p_hyp_train.jsonl").exists()
    lines.append(f"hyp probe p_hyp_train.jsonl: {'OK' if probe else 'FAIL'}")
    ok = ok and probe
    print("\n".join(lines))
    print("G1:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
