"""r_acc, r_fmt, r_struct dạng hàm thuần.
"""
from formats.align import struct_ok
from formats.parser import Parsed, parse


def r_acc(p: Parsed, gold: str) -> float:
    return 1.0 if p.format_ok and p.label == gold else 0.0


def r_fmt(p: Parsed) -> float:
    return 1.0 if p.format_ok else 0.0


def r_struct(p: Parsed, fmt: str, premise: str, hypothesis: str) -> float:
    if fmt != "align" or not p.format_ok:
        return 0.0
    return 1.0 if struct_ok(p.align_lines, premise, hypothesis, p.label) else 0.0


def total_reward(text: str, fmt: str, guess: bool, gold: str, premise: str, hypothesis: str) -> float:
    p = parse(text, fmt, guess)
    a = r_acc(p, gold)
    return a + 0.2 * r_fmt(p) + 0.3 * a * r_struct(p, fmt, premise, hypothesis)
