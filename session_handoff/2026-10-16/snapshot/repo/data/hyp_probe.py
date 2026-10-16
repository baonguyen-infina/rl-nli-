"""CafeBERT hypothesis-only: 5-fold cross-fitting trên train -> p_hyp; model full-train -> nhãn hyp-easy/hyp-hard cho test ViANLI và ViNLI.
"""
import numpy as np

from common.io import ARTIFACTS, load_config, write_jsonl


def fold_indices(n: int, k: int, seed: int):
    perm = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(perm, k)
    return [(np.concatenate([parts[j] for j in range(k) if j != i]), parts[i]) for i in range(k)]


def cross_fit(train_rows: list[dict], cfg: dict, seed: int = 0) -> np.ndarray:
    from train.encoder import train_and_predict
    p = np.zeros((len(train_rows), 3))
    for tr, va in fold_indices(len(train_rows), cfg["encoder"]["folds"], seed):
        p[va] = train_and_predict([train_rows[i] for i in tr], [train_rows[i] for i in va],
                                  cfg, mode="hyp", seed=seed)
    return p


def main() -> None:
    # Chỉ phần cross-fitting trên train. Gán nhãn hyp-easy/hard cho test chờ G4 (test chưa được chạm).
    from data.prepare import get_rows
    cfg = load_config("B1")
    rows = get_rows("train")
    p = cross_fit(rows, cfg)
    write_jsonl(ARTIFACTS / "hyp_probe" / "p_hyp_train.jsonl",
                [{"id": r["id"], "p_hyp": p[i].tolist()} for i, r in enumerate(rows)])
    print("OK", p.shape)


if __name__ == "__main__":
    main()
