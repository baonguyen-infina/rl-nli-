"""Prompt (tiếng Việt) cho 3 format label / cot / align, có/không <guess>; kèm 1 ví dụ format tự viết.
"""
EXAMPLE_PREMISE = "Hà Nội là thủ đô của Việt Nam và có hơn tám triệu dân."
EXAMPLE_HYPOTHESIS = "Hà Nội là thủ đô của Việt Nam."

_ALIGN_BLOCK = ('<align>\nH1: "Hà Nội là thủ đô của Việt Nam" <-> '
                '"Hà Nội là thủ đô của Việt Nam" | match\n</align>')

EXAMPLE_OUTPUTS = {
    ("label", False): "<answer>entailment</answer>",
    ("cot", False): ("<think>Premise nêu rõ Hà Nội là thủ đô của Việt Nam, "
                     "trùng với hypothesis.</think><answer>entailment</answer>"),
    ("cot", True): ("<guess>entailment</guess><think>Premise nêu rõ Hà Nội là thủ đô của "
                    "Việt Nam, trùng với hypothesis.</think><answer>entailment</answer>"),
    ("align", False): _ALIGN_BLOCK + "<answer>entailment</answer>",
    ("align", True): "<guess>entailment</guess>" + _ALIGN_BLOCK + "<answer>entailment</answer>",
}

_INTRO = (
    "Bạn là hệ thống suy luận ngôn ngữ tự nhiên. Cho một tiền đề (premise) và một giả thuyết "
    "(hypothesis) bằng tiếng Việt, hãy xác định quan hệ của giả thuyết với tiền đề:\n"
    "- entailment: giả thuyết suy ra đúng từ tiền đề.\n"
    "- neutral: tiền đề không đủ thông tin để kết luận.\n"
    "- contradiction: giả thuyết mâu thuẫn với tiền đề.\n"
)

_FORMAT_HELP = {
    "label": "Chỉ trả lời đúng một thẻ: <answer>NHÃN</answer>, không viết gì thêm.",
    "cot": ("Trước tiên lập luận ngắn gọn trong <think>...</think>, sau đó trả lời trong "
            "<answer>NHÃN</answer>. Không viết gì ngoài các thẻ."),
    "align": (
        "Trước tiên căn chỉnh từng đoạn của hypothesis với premise trong khối <align>...</align>, "
        "sau đó trả lời trong <answer>NHÃN</answer>. Mỗi dòng (1 đến 6 dòng) có dạng:\n"
        'H1: "<đoạn hypothesis>" <-> "<đoạn premise>" | match\n'
        'H2: "<đoạn hypothesis>" <-> ∅ | missing\n'
        'H3: "<đoạn hypothesis>" <-> "<đoạn premise>" | conflict\n'
        "Các đoạn phải được chép nguyên văn từ hypothesis và premise; các đoạn hypothesis cần "
        "phủ ít nhất một nửa hypothesis. match: đoạn có trong premise; missing: premise không "
        "nhắc tới (dùng ∅); conflict: đoạn mâu thuẫn với premise. Có conflict thì nhãn là "
        "contradiction; không conflict mà có missing thì nhãn là neutral; còn lại là entailment. "
        "Không viết gì ngoài các thẻ."),
}

_GUESS_HELP = ("Ngay từ đầu, hãy đoán nhanh nhãn trong <guess>NHÃN</guess>, rồi mới viết phần còn lại.")


def build_prompt(premise: str, hypothesis: str, fmt: str, guess: bool = False) -> str:
    if fmt not in _FORMAT_HELP:
        raise ValueError(f"format không hợp lệ: {fmt!r}")
    if guess and fmt == "label":
        raise ValueError("guess chỉ dùng cho cot/align")
    parts = [_INTRO, _FORMAT_HELP[fmt]]
    if guess:
        parts.append(_GUESS_HELP)
    parts.append(
        f"\nVí dụ định dạng:\nPremise: {EXAMPLE_PREMISE}\nHypothesis: {EXAMPLE_HYPOTHESIS}\n"
        f"Trả lời: {EXAMPLE_OUTPUTS[(fmt, guess)]}\n"
    )
    parts.append(f"Bây giờ hãy làm với cặp sau.\nPremise: {premise}\nHypothesis: {hypothesis}\nTrả lời:")
    return "\n".join(parts)


def build_messages(premise: str, hypothesis: str, fmt: str, guess: bool = False) -> list[dict]:
    return [{"role": "user", "content": build_prompt(premise, hypothesis, fmt, guess)}]
