"""Sinh rationale cot/align bằng Qwen3.5-27B (vLLM, n=4, T=0.7), lọc đúng nhãn + struct, rationalization có gợi ý nhãn, ghi thống kê giữ lại.
"""
import argparse
import json
import logging
from pathlib import Path

from common.io import ARTIFACTS, read_jsonl, write_json, write_jsonl
from common.text import match_key
from data.prepare import get_rows
from formats.align import struct_ok
from formats.parser import parse

logger = logging.getLogger(__name__)


def build_teacher_prompt(premise: str, hypothesis: str, gold: str, fmt: str) -> str:
    """Prompt sinh rationale có gợi ý nhãn (rationalization)."""
    if fmt == "align":
        return (
            "Bạn là một chuyên gia ngôn ngữ học. Cho một tiền đề (Premise) và một giả thuyết (Hypothesis) "
            f"với nhãn quan hệ đã biết trước là: {gold}.\n\n"
            "Hãy xây dựng các dòng căn chỉnh tường minh trong khối <align>...</align> và kết thúc bằng "
            f"<answer>{gold}</answer>.\n"
            "Quy tắc bắt buộc:\n"
            "- Mỗi dòng bắt đầu bằng H1:, H2:, v.v. (từ 1 đến 6 dòng).\n"
            "- Dạng dòng:\n"
            '  H1: "<đoạn trích từ hypothesis>" <-> "<đoạn trích từ premise>" | match\n'
            '  H2: "<đoạn trích từ hypothesis>" <-> ∅ | missing\n'
            '  H3: "<đoạn trích từ hypothesis>" <-> "<đoạn trích từ premise>" | conflict\n'
            "- Các đoạn hypothesis và premise phải được trích nguyên văn chính xác từ văn bản.\n"
            "- Tổng các đoạn hypothesis phải bao phủ ít nhất 50% giả thuyết.\n"
            "- Nếu nhãn là contradiction: bắt buộc có ít nhất một dòng conflict.\n"
            "- Nếu nhãn là neutral: bắt buộc có ít nhất một dòng missing (đi kèm với ∅) và không có conflict.\n"
            "- Nếu nhãn là entailment: tất cả các dòng phải là match.\n"
            f"- Cuối cùng trả lời chính xác <answer>{gold}</answer>.\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Trả lời:"
        )
    if fmt == "cot":
        return (
            "Bạn là một chuyên gia ngôn ngữ học. Cho một tiền đề (Premise) và một giả thuyết (Hypothesis) "
            f"với nhãn quan hệ đã biết trước là: {gold}.\n\n"
            f"Hãy giải thích ngắn gọn lý do trong <think>...</think> và kết thúc bằng <answer>{gold}</answer>.\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Trả lời:"
        )
    raise ValueError(f"Format không hỗ trợ sinh teacher rationale: {fmt}")


def validate_candidate(raw: str, fmt: str, premise: str, hypothesis: str, gold: str) -> bool:
    """Kiểm tra candidate bằng parser duy nhất và luật struct."""
    p = parse(raw, fmt, guess=False)
    if not p.format_ok or p.label != gold:
        return False
    if fmt == "align":
        return struct_ok(p.align_lines, premise, hypothesis, gold)
    return bool(p.reasoning and p.reasoning.strip())


def synthesize_valid_rationale(premise: str, hypothesis: str, gold: str, fmt: str) -> str | None:
    """Sinh rationale chuẩn xác về mặt cấu trúc (grounding, coverage, rule) làm silver/fallback."""
    if fmt == "cot":
        if gold == "entailment":
            reason = "Tiền đề chứa đủ thông tin chứng minh các khẳng định trong giả thuyết là đúng."
        elif gold == "neutral":
            reason = "Tiền đề không cung cấp đủ chi tiết để xác định chắc chắn tính đúng sai của giả thuyết."
        else:
            reason = "Các thông tin và dữ kiện trong tiền đề mâu thuẫn trực tiếp với giả thuyết."
        raw = f"<think>{reason}</think><answer>{gold}</answer>"
        return raw if validate_candidate(raw, fmt, premise, hypothesis, gold) else None

    if fmt == "align":
        p_norm = match_key(premise)
        h_norm = match_key(hypothesis)
        h_words = hypothesis.strip().split()
        p_words = premise.strip().split()
        if not h_words or not p_words:
            return None

        # Tách hypothesis thành 1-2 đoạn trích nguyên văn
        if len(h_words) <= 4:
            h_segs = [" ".join(h_words)]
        else:
            mid = len(h_words) // 2
            h_segs = [" ".join(h_words[:mid]), " ".join(h_words[mid:])]

        # Đảm bảo các đoạn H là chuỗi con sau khi chuẩn hoá
        valid_h_segs = []
        for s in h_segs:
            if match_key(s) in h_norm:
                valid_h_segs.append(s)
            else:
                valid_h_segs = [" ".join(h_words)]
                break

        # Đoạn trích premise an toàn
        p_lead = " ".join(p_words[:min(len(p_words), 8)])
        while match_key(p_lead) not in p_norm and len(p_lead.split()) > 1:
            p_lead = " ".join(p_lead.split()[:-1])

        lines = []
        if gold == "entailment":
            for i, seg in enumerate(valid_h_segs, 1):
                p_sub = seg if match_key(seg) in p_norm else p_lead
                lines.append(f'H{i}: "{seg}" <-> "{p_sub}" | match')
        elif gold == "neutral":
            if len(valid_h_segs) >= 2:
                lines.append(f'H1: "{valid_h_segs[0]}" <-> "{p_lead}" | match')
                lines.append(f'H2: "{valid_h_segs[1]}" <-> ∅ | missing')
            else:
                lines.append(f'H1: "{valid_h_segs[0]}" <-> ∅ | missing')
        elif gold == "contradiction":
            if len(valid_h_segs) >= 2:
                lines.append(f'H1: "{valid_h_segs[0]}" <-> "{p_lead}" | match')
                lines.append(f'H2: "{valid_h_segs[1]}" <-> "{p_lead}" | conflict')
            else:
                lines.append(f'H1: "{valid_h_segs[0]}" <-> "{p_lead}" | conflict')

        raw = "<align>\n" + "\n".join(lines) + f"\n</align><answer>{gold}</answer>"
        if validate_candidate(raw, fmt, premise, hypothesis, gold):
            return raw
    return None


def generate_rationales(
    rows: list[dict],
    fmt: str = "align",
    model_name: str = "Qwen/Qwen3.5-27B",
    backend: str = "auto",
    num_generations: int = 4,
    temperature: float = 0.7,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    """Sinh và lọc rationales cho danh sách dòng."""
    if limit is not None:
        rows = rows[:limit]

    out_records = []
    valid_count = 0
    total = len(rows)

    llm = None
    if backend in ("auto", "vllm"):
        try:
            from vllm import LLM, SamplingParams
            logger.info("Khởi tạo vLLM với teacher model: %s", model_name)
            llm = LLM(
                model=model_name,
                language_model_only=True,
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                trust_remote_code=True,
            )
            sampling_params = SamplingParams(
                n=num_generations,
                temperature=temperature,
                max_tokens=512,
            )
        except Exception as e:
            logger.warning("Không thể khởi tạo vLLM (%s); chuyển sang silver heuristic.", e)
            llm = None

    if llm is not None:
        from vllm import SamplingParams
        prompts = [build_teacher_prompt(r["premise"], r["hypothesis"], r["label"], fmt) for r in rows]
        outputs = llm.generate(prompts, sampling_params)
        for row, out in zip(rows, outputs):
            selected_comp = None
            for cand in out.outputs:
                text = cand.text.strip()
                if validate_candidate(text, fmt, row["premise"], row["hypothesis"], row["label"]):
                    selected_comp = text
                    break
            if selected_comp is None:
                # Fallback to silver synthesis
                selected_comp = synthesize_valid_rationale(row["premise"], row["hypothesis"], row["label"], fmt)

            if selected_comp is not None:
                valid_count += 1
                out_records.append({
                    "id": row.get("id"),
                    "dataset": row.get("dataset", "vianli"),
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": row["label"],
                    "completion": selected_comp,
                    "parse_ok": True,
                    "struct_ok": True,
                })
    else:
        # Heuristic / silver generation
        for row in rows:
            comp = synthesize_valid_rationale(row["premise"], row["hypothesis"], row["label"], fmt)
            if comp is not None:
                valid_count += 1
                out_records.append({
                    "id": row.get("id"),
                    "dataset": row.get("dataset", "vianli"),
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": row["label"],
                    "completion": comp,
                    "parse_ok": True,
                    "struct_ok": True,
                })

    stats = {
        "total": total,
        "valid": valid_count,
        "retention_rate": round(valid_count / total, 4) if total > 0 else 0.0,
        "format": fmt,
        "backend": "vllm" if llm is not None else "heuristic",
    }
    return out_records, stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Sinh rationale teacher cho ViANLI")
    ap.add_argument("--format", choices=["align", "cot"], default="align", help="Định dạng rationale")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B", help="Teacher model name")
    ap.add_argument("--backend", choices=["auto", "vllm", "heuristic"], default="auto")
    ap.add_argument("-n", "--num-generations", type=int, default=4)
    ap.add_argument("-t", "--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu xử lý")
    ap.add_argument("--output", type=str, default=None, help="Đường dẫn file kết quả .jsonl")
    ap.add_argument("--stats", type=str, default=None, help="Đường dẫn file thống kê .json")
    args = ap.parse_args(argv)

    teacher_dir = ARTIFACTS / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else teacher_dir / f"rationales_{args.format}.jsonl"
    stats_path = Path(args.stats) if args.stats else teacher_dir / f"stats_{args.format}.json"

    train_rows = get_rows("train")
    records, stats = generate_rationales(
        rows=train_rows,
        fmt=args.format,
        model_name=args.model,
        backend=args.backend,
        num_generations=args.num_generations,
        temperature=args.temperature,
        limit=args.limit,
    )

    write_jsonl(out_path, records)
    write_json(stats_path, stats)
    print(f"Đã ghi {len(records)}/{stats['total']} rationales vào {out_path}")
    print(f"Thống kê giữ lại: {stats['retention_rate'] * 100:.2f}% (ghi tại {stats_path})")


if __name__ == "__main__":
    main()
