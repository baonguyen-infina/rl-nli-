"""Bộ nhãn chung (entailment, neutral, contradiction) và hàm chuẩn hoá nhãn của dataset.
"""
LABELS = ("entailment", "neutral", "contradiction")
INTENDED_CODE = {"11": "entailment", "21": "neutral", "31": "contradiction"}


def normalize_label(x: str) -> str:
    y = str(x).strip().lower()
    if y not in LABELS:
        raise ValueError(f"nhãn không hợp lệ: {x!r}")
    return y
