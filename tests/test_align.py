from formats.align import (AlignLine, parse_align_block, rule_label,
                           is_grounded, coverage, struct_ok)

PREM = "Hà Nội là thủ đô của Việt Nam và có hơn tám triệu dân."
HYP = "Hà Nội là thủ đô của Việt Nam."


def test_parse_valid_block():
    block = ('H1: "Hà Nội là thủ đô" <-> "Hà Nội là thủ đô" | match\n'
             'H2: "của Việt Nam" <-> ∅ | missing')
    lines = parse_align_block(block)
    assert lines == [AlignLine(1, "Hà Nội là thủ đô", "Hà Nội là thủ đô", "match"),
                     AlignLine(2, "của Việt Nam", None, "missing")]


def test_parse_rejects_bad_blocks():
    assert parse_align_block("") is None
    # missing must be paired with ∅
    assert parse_align_block('H1: "a" <-> "b" | missing') is None
    # match must not use ∅
    assert parse_align_block('H1: "a" <-> ∅ | match') is None
    # wrong numbering
    assert parse_align_block('H2: "a" <-> "b" | match') is None
    # unknown relation
    assert parse_align_block('H1: "a" <-> "b" | maybe') is None
    # more than 6 lines
    seven = "\n".join(f'H{i}: "a" <-> "b" | match' for i in range(1, 8))
    assert parse_align_block(seven) is None
    six = "\n".join(f'H{i}: "a" <-> "b" | match' for i in range(1, 7))
    assert parse_align_block(six) is not None


def test_rule_label_priority():
    m = AlignLine(1, "a", "b", "match")
    x = AlignLine(2, "a", None, "missing")
    c = AlignLine(3, "a", "b", "conflict")
    assert rule_label([m]) == "entailment"
    assert rule_label([m, x]) == "neutral"
    assert rule_label([m, x, c]) == "contradiction"


def test_grounding():
    ok = [AlignLine(1, "hà nội là thủ đô", "HÀ NỘI  là thủ đô", "match")]
    assert is_grounded(ok, PREM, HYP)
    bad_h = [AlignLine(1, "Sài Gòn", "Hà Nội", "match")]
    assert not is_grounded(bad_h, PREM, HYP)
    bad_p = [AlignLine(1, "Hà Nội", "Đà Nẵng", "match")]
    assert not is_grounded(bad_p, PREM, HYP)
    empty = [AlignLine(1, "  ", None, "missing")]
    assert not is_grounded(empty, PREM, HYP)


def test_coverage_threshold():
    hyp = "một hai ba bốn năm sáu bảy tám chín mười"  # 10 syllables
    half = [AlignLine(1, "một hai ba bốn năm", None, "missing")]
    less = [AlignLine(1, "một hai ba bốn", None, "missing")]
    assert coverage(half, hyp) == 0.5
    assert coverage(less, hyp) == 0.4
    # overlapping segments are not double counted
    ov = [AlignLine(1, "một hai ba", None, "missing"), AlignLine(2, "hai ba bốn", None, "missing")]
    assert coverage(ov, hyp) == 0.4


def test_struct_ok():
    lines = [AlignLine(1, "Hà Nội là thủ đô của Việt Nam", "Hà Nội là thủ đô của Việt Nam", "match")]
    assert struct_ok(lines, PREM, HYP, "entailment")
    assert not struct_ok(lines, PREM, HYP, "neutral")  # rule disagrees with answer
    short = [AlignLine(1, "Hà Nội", "Hà Nội", "match")]
    assert not struct_ok(short, PREM, HYP, "entailment")  # coverage 2/8 < 0.5
