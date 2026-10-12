"""Chuẩn hoá NFC + khoảng trắng, khoá so khớp (casefold) cho grounding, tách âm tiết để tính coverage.
"""
import re
import unicodedata

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s)).strip()


def match_key(s: str) -> str:
    return normalize(s).casefold()


def syllables(s: str) -> list[str]:
    return _TOKEN.findall(match_key(s))
