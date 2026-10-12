from train.sft import pick_best


def test_pick_best_max_accuracy():
    assert pick_best({"checkpoint-100": 0.5, "checkpoint-200": 0.7, "checkpoint-300": 0.6}) == "checkpoint-200"


def test_pick_best_tie_prefers_latest():
    assert pick_best({"checkpoint-100": 0.7, "checkpoint-200": 0.7}) == "checkpoint-200"
