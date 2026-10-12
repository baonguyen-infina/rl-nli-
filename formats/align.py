"""Parse dòng align, luật tổng hợp nhãn (conflict > missing > match), kiểm grounding, coverage >= 0.5, struct_ok.
"""
import re
from dataclasses import dataclass

from common.text import match_key, syllables

LINE_RE = re.compile(
    r'^H(\d+):\s*"(.+?)"\s*<->\s*(?:"(.+)"|∅)\s*\|\s*(match|missing|conflict)$'
)
COVERAGE_MIN = 0.5


@dataclass(frozen=True)
class AlignLine:
    idx: int
    h: str
    p: str | None
    rel: str


def parse_align_block(block: str) -> list[AlignLine] | None:
    raw = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if not 1 <= len(raw) <= 6:
        return None
    out = []
    for i, ln in enumerate(raw, 1):
        m = LINE_RE.match(ln)
        if not m or int(m.group(1)) != i:
            return None
        h, p, rel = m.group(2), m.group(3), m.group(4)
        if rel == "missing" and p is not None:
            return None
        if rel != "missing" and p is None:
            return None
        out.append(AlignLine(i, h, p, rel))
    return out


def rule_label(lines: list[AlignLine]) -> str:
    rels = {ln.rel for ln in lines}
    if "conflict" in rels:
        return "contradiction"
    if "missing" in rels:
        return "neutral"
    return "entailment"


def is_grounded(lines: list[AlignLine], premise: str, hypothesis: str) -> bool:
    prem, hyp = match_key(premise), match_key(hypothesis)
    for ln in lines:
        h = match_key(ln.h)
        if not h or h not in hyp:
            return False
        if ln.p is not None:
            p = match_key(ln.p)
            if not p or p not in prem:
                return False
    return True


def coverage(lines: list[AlignLine], hypothesis: str) -> float:
    toks = syllables(hypothesis)
    if not toks:
        return 0.0
    covered: set[int] = set()
    for ln in lines:
        seg = syllables(ln.h)
        n = len(seg)
        if n == 0:
            continue
        for s in range(len(toks) - n + 1):
            if toks[s:s + n] == seg:
                covered.update(range(s, s + n))
                break
    return len(covered) / len(toks)


def struct_ok(lines: list[AlignLine], premise: str, hypothesis: str, answer_label: str) -> bool:
    return (
        is_grounded(lines, premise, hypothesis)
        and coverage(lines, hypothesis) >= COVERAGE_MIN
        and rule_label(lines) == answer_label
    )
