"""Parser chặt cho output của từng format -> label, guess, phần suy luận, các dòng align, cờ format_ok. Dùng chung cho reward và eval.
"""
import re
from dataclasses import dataclass

from formats.align import AlignLine, parse_align_block

_L = "entailment|neutral|contradiction"
_ANS = rf"<answer>(?P<answer>{_L})</answer>"
_GUESS = rf"<guess>(?P<guess>{_L})</guess>\s*"
_TAGS = {
    "label": ["answer"],
    "cot": ["think", "answer"],
    "align": ["align", "answer"],
}


@dataclass
class Parsed:
    label: str | None = None
    guess: str | None = None
    reasoning: str | None = None
    align_lines: list[AlignLine] | None = None
    format_ok: bool = False


def _pattern(fmt: str, guess: bool) -> re.Pattern:
    head = _GUESS if guess else ""
    if fmt == "label":
        body = _ANS
    elif fmt == "cot":
        body = rf"<think>(?P<think>.+?)</think>\s*{_ANS}"
    else:
        body = rf"<align>\s*(?P<block>.+?)\s*</align>\s*{_ANS}"
    return re.compile(head + body, re.DOTALL)


def parse(text: str, fmt: str, guess: bool = False) -> Parsed:
    if fmt not in _TAGS:
        raise ValueError(f"format không hợp lệ: {fmt!r}")
    if guess and fmt == "label":
        raise ValueError("guess chỉ dùng cho cot/align")
    text = (text or "").strip()
    tags = _TAGS[fmt] + (["guess"] if guess else [])
    for t in tags:
        if text.count(f"<{t}>") != 1 or text.count(f"</{t}>") != 1:
            return Parsed()
    if not guess and "<guess>" in text:
        return Parsed()
    m = _pattern(fmt, guess).fullmatch(text)
    if not m:
        return Parsed()
    gd = m.groupdict()
    lines = None
    reasoning = None
    if fmt == "cot":
        reasoning = gd["think"].strip()
        if not reasoning:
            return Parsed()
    elif fmt == "align":
        reasoning = gd["block"]
        lines = parse_align_block(gd["block"])
        if lines is None:
            return Parsed()
    return Parsed(label=gd["answer"], guess=gd.get("guess"), reasoning=reasoning,
                  align_lines=lines, format_ok=True)
