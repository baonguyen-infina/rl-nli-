"""Kiểm cổng G1: E3 > zero-shot (E0–E2) trên dev và parse E3 >= 99%; đủ E0–E3, B1, hyp probe."""
import sys

from common.io import ARTIFACTS, RESULTS, read_json

REQUIRED = ["E0", "E1", "E2", "E3", "B1"]


def evaluate_gate(metrics: dict, probe_complete: bool = False) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for rid in REQUIRED:
        if rid not in metrics:
            lines.append(f"thiếu kết quả {rid}")
            ok = False
    if not ok:
        return False, lines
    zs = max(metrics[r]["macro_f1"] for r in ("E0", "E1", "E2"))
    e3 = metrics["E3"]
    better = e3["macro_f1"] > zs
    parse_ok = e3["parse_rate"] >= 0.99
    lines.append(f"E3 Macro-F1 {e3['macro_f1']:.4f} vs best ZS {zs:.4f}: {'OK' if better else 'FAIL'}")
    lines.append(f"E3 parse {e3['parse_rate']:.4f} >= 0.99: {'OK' if parse_ok else 'FAIL'}")
    lines.append(f"B1 Macro-F1 {metrics['B1']['macro_f1']:.4f} (tham chiếu)")
    lines.append(f"hypothesis probe: {'OK' if probe_complete else 'thiếu'}")
    return better and parse_ok and probe_complete, lines


def main() -> None:
    metrics = {}
    for rid in ("E0", "E1", "E2", "E3", "B1"):
        for d in sorted(RESULTS.glob(f"{rid}_*_s0")):
            f = d / "metrics_dev.json"
            if f.exists():
                metrics[rid] = read_json(f)
    probe = (ARTIFACTS / "hyp_probe" / "p_hyp_train.jsonl").exists()
    ok, lines = evaluate_gate(metrics, probe_complete=probe)
    print("\n".join(lines))
    print("G1:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
