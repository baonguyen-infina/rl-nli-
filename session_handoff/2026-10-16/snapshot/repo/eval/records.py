"""Dựng một dòng predictions_<split>.jsonl (LLM và encoder) theo đúng schema của AGENTS.md.
"""
from formats.align import coverage, is_grounded, rule_label
from formats.parser import parse


def build_record(row: dict, raw_output: str, fmt: str, guess: bool, n_tokens: int) -> dict:
    p = parse(raw_output, fmt, guess)
    rule = grounded = cov = None
    if fmt == "align" and p.format_ok:
        rule = rule_label(p.align_lines)
        grounded = is_grounded(p.align_lines, row["premise"], row["hypothesis"])
        cov = coverage(p.align_lines, row["hypothesis"])
    return {"id": row["id"], "dataset": row["dataset"], "topic": row["topic"],
            "gold": row["label"], "pred": p.label, "guess": p.guess,
            "parse_ok": p.format_ok, "rule_label": rule, "grounded": grounded,
            "coverage": cov, "n_tokens": n_tokens, "raw_output": raw_output}


def encoder_record(row: dict, pred: str, n_tokens: int = 0) -> dict:
    return {"id": row["id"], "dataset": row["dataset"], "topic": row["topic"],
            "gold": row["label"], "pred": pred, "guess": None, "parse_ok": True,
            "rule_label": None, "grounded": None, "coverage": None,
            "n_tokens": n_tokens, "raw_output": ""}
