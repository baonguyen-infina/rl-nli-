"""Bọc thành reward_funcs cho TRL GRPOTrainer (tên r_acc, r_fmt, r_acc_struct; trọng số 1.0 / 0.2 / 0.3).
"""
from formats.parser import parse
from rewards.core import r_acc, r_fmt, r_struct

REWARD_WEIGHTS = [1.0, 0.2, 0.3]


def completion_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1]["content"]
    raise TypeError(f"completion không hỗ trợ: {type(c)}")


def make_reward_funcs(fmt: str, guess: bool = False):
    def _parsed(completions):
        return [parse(completion_text(c), fmt, guess) for c in completions]

    def f_acc(prompts, completions, **kwargs):
        return [r_acc(p, g) for p, g in zip(_parsed(completions), kwargs["gold"])]
    f_acc.__name__ = "r_acc"

    def f_fmt(prompts, completions, **kwargs):
        return [r_fmt(p) for p in _parsed(completions)]
    f_fmt.__name__ = "r_fmt"

    def f_struct(prompts, completions, **kwargs):
        out = []
        for p, g, pr, hy in zip(_parsed(completions), kwargs["gold"],
                                kwargs["premise"], kwargs["hypothesis"]):
            out.append(r_acc(p, g) * r_struct(p, fmt, pr, hy))
        return out
    f_struct.__name__ = "r_acc_struct"

    return [f_acc, f_fmt, f_struct]
