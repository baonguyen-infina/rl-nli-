"""Tải ViANLI (uitnlp/ViANLI), tách uid -> premise_id, intended_label, topic; kiểm số đếm 8012/1000/1000; tạo dev-mini 300. Nạp ViNLI (bỏ OTHER), ViMedNLI từ bản local. KHÔNG đưa uid vào prompt.
"""
import os
import random
import re

from common.io import ARTIFACTS, read_json, read_jsonl, write_json, write_jsonl
from common.labels import INTENDED_CODE, normalize_label

UID_RE = re.compile(r"^uit_Adver_(.+)_(\d+)_(11|21|31)_(0[1-9]|1[0-3])$")
EXPECTED_COUNTS = {"train": 8012, "dev": 1000, "test": 1000}
DATA_DIR = ARTIFACTS / "data"


def parse_uid(uid: str) -> dict:
    m = UID_RE.match(uid)
    if not m:
        raise ValueError(f"uid không khớp định dạng ViANLI: {uid!r}")
    return {"premise_id": m.group(1), "k": m.group(2),
            "intended_label": INTENDED_CODE[m.group(3)], "topic": m.group(4)}


def prepare_rows(raw: list[dict], dataset: str = "vianli") -> list[dict]:
    rows = []
    for r in raw:
        u = parse_uid(r["uid"])
        rows.append({
            "id": r["uid"], "dataset": dataset,
            "premise": r["premise"], "hypothesis": r["hypothesis"],
            "label": normalize_label(r["label"]),
            "topic": u["topic"], "premise_id": u["premise_id"],
            "intended_label": u["intended_label"],
        })
    return rows


def check_counts(counts: dict[str, int]) -> None:
    for split, n in counts.items():
        assert n == EXPECTED_COUNTS[split], f"{split}: {n} != {EXPECTED_COUNTS[split]}"


def make_dev_mini(rows: list[dict], n: int = 300, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        groups.setdefault(r["label"], []).append(r["id"])
    total = len(rows)
    quota = {lab: int(n * len(ids) / total) for lab, ids in groups.items()}
    while sum(quota.values()) < n:
        lab = max(groups, key=lambda l: len(groups[l]) - quota[l])
        quota[lab] += 1
    chosen = []
    for lab in sorted(groups):
        ids = list(groups[lab])
        rng.shuffle(ids)
        chosen.extend(ids[:quota[lab]])
    return sorted(chosen)


def _guard(split: str, allow_test: bool) -> None:
    if split == "test" and not (allow_test and os.environ.get("G4_UNLOCK") == "1"):
        raise PermissionError("test split bị khoá đến G4 (cần allow_test=True và G4_UNLOCK=1)")


def load_raw(split: str, allow_test: bool = False) -> list[dict]:
    _guard(split, allow_test)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="uitnlp/ViANLI", filename=f"vianli_{split}.jsonl",
                           repo_type="dataset")
    return read_jsonl(path)


def get_rows(split: str, allow_test: bool = False) -> list[dict]:
    _guard(split, allow_test)
    if split == "dev_mini":
        ids = set(read_json(DATA_DIR / "dev_mini_ids.json"))
        return [r for r in read_jsonl(DATA_DIR / "vianli_dev.jsonl") if r["id"] in ids]
    return read_jsonl(DATA_DIR / f"vianli_{split}.jsonl")


def main() -> None:
    counts = {}
    for split in ("train", "dev"):  # test không được chạm trước G4
        rows = prepare_rows(load_raw(split))
        counts[split] = len(rows)
        write_jsonl(DATA_DIR / f"vianli_{split}.jsonl", rows)
    check_counts(counts)
    dev = read_jsonl(DATA_DIR / "vianli_dev.jsonl")
    write_json(DATA_DIR / "dev_mini_ids.json", make_dev_mini(dev, n=300, seed=42))
    print("OK", counts)


if __name__ == "__main__":
    main()
