"""Step 7 of the pipeline: EVALUATE.

Runs every question in the questions file through the same ask_question()
that ask.py uses, scores each answer, saves one row per question to
workspace/eval/results.csv, and prints accuracy for each category.

How answers are scored:
- numeric:      correct if any number in the answer is within a small tolerance
                (evaluation.numeric_tolerance in config.yaml) of the reference value.
                "$416.2 billion", "$416,161 million" and "416,161" all count for
                416,161,000,000. A diluted EPS must be within half a cent.
- text:         correct if every key fact from the question file appears in the answer.
- unanswerable: correct if the answer says it doesn't know (or the pipeline refused
                because the best chunk scored below abstention.min_score).

The questions file is a JSON list; see data/eval/questions.json for the format.
"""

import csv
import json
import re
import time

from interfaces import PipelineError
from pipeline import ask_question

CATEGORIES = ["numeric", "text", "unanswerable"]

# A number such as 416,161 or 94.83, optionally followed by a size word or letter.
NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*((?i:trillion|billion|million|thousand|bn|mn))\b|([MBK])\b)?"
)
SCALES = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3,
          "bn": 1e9, "mn": 1e6, "m": 1e6, "b": 1e9, "k": 1e3}

# Ways of saying "I don't know".
ABSTAIN = re.compile(
    r"don't know|do not know|could not find|couldn't find|cannot find|can't find|unable to (?:find|determine|answer)"
    r"|not (?:contain|provide|mention|include)|no information|not (?:available )?in the sources",
    re.IGNORECASE,
)


def plain(text):
    """Lower-case text with curly quotes straightened and [1]-style citations removed."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\[\d+\]", " ", text).lower()


def numbers_in(answer, per_share):
    """List every value the answer could be stating, as plain dollars (or dollars per share)."""
    values = []
    for match in NUMBER.finditer(re.sub(r"\[\d+\]", " ", answer)):
        if answer[match.end():match.end() + 1] == "%":
            continue  # a percentage, not an amount
        x = float(match.group(1).replace(",", ""))
        word = match.group(2) or match.group(3)
        if per_share:
            values.append(x)
        elif word:
            values.append(x * SCALES[word.lower()])
        else:
            values.extend([x, x * 1e6])  # filings state amounts in millions, so "416,161" means that
    return values


def score_numeric(question, answer, tolerance):
    expected = question["expected_value"]
    per_share = question["unit"] == "USD/shares"
    for value in numbers_in(answer, per_share):
        if per_share:
            if abs(value - expected) <= 0.005 + 1e-9:
                return True
        elif abs(value - expected) <= tolerance * abs(expected):
            return True
    return False


def score_text(question, answer):
    text = plain(answer)
    return all(any(option in text for option in options) for options in question["key_facts"])


def score_unanswerable(result):
    return result["abstained"] or bool(ABSTAIN.search(plain(result["answer"])))


def expected_text(question):
    if question["category"] == "numeric":
        return f"{question['expected_value']:,} {question['unit']}"
    if question["category"] == "text":
        return question["reference_answer"]
    return "I don't know"


def run_evaluation(config, retriever, language_model, category=None):
    """Run the test set. Returns {category: (correct, total), ..., "overall": (correct, total)}."""
    questions_file = config.questions_file
    if not questions_file.exists():
        raise PipelineError(
            f"{questions_file} not found. For the SEC filings run: py src/build_eval_questions.py --email you@example.com"
        )
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    if category:
        questions = [q for q in questions if q["category"] == category]
    tolerance = config.section("evaluation")["numeric_tolerance"]

    rows = []
    for number, q in enumerate(questions, start=1):
        started = time.time()
        result = ask_question(q["question"], retriever, language_model, config)
        seconds = time.time() - started

        if q["category"] == "numeric":
            correct = score_numeric(q, result["answer"], tolerance)
        elif q["category"] == "text":
            correct = score_text(q, result["answer"])
        else:
            correct = score_unanswerable(result)

        rows.append({
            "id": q["id"], "category": q["category"], "ticker": q.get("ticker") or "",
            "question": q["question"], "expected": expected_text(q),
            "answer": " ".join(result["answer"].split()),
            "correct": int(correct), "refused_by_threshold": int(result["abstained"]),
            "top_score": round(result["top_score"], 3),
            "top_chunk": result["sources"][0]["id"], "seconds": round(seconds, 1),
        })
        print(f"[{number:2d}/{len(questions)}] {'PASS' if correct else 'FAIL'}  {q['id']:8s} {q['question'][:70]}", flush=True)

    out = config.results_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows symbols correctly
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lm = config.section("language_model")
    print(f"\nResults (language model {lm.get('name', lm['type'])}, retriever {config.section('retriever')['type']}, "
          f"min_score {config.section('abstention')['min_score']}, numeric tolerance {tolerance:.1%})")
    print(f"{'Category':14s}{'Correct':>8s}{'Total':>7s}{'Accuracy':>10s}")
    summary = {}
    for name in CATEGORIES:
        group = [r for r in rows if r["category"] == name]
        if group:
            right = sum(r["correct"] for r in group)
            summary[name] = (right, len(group))
            print(f"{name:14s}{right:8d}{len(group):7d}{right / len(group):10.1%}")
    right = sum(r["correct"] for r in rows)
    summary["overall"] = (right, len(rows))
    print(f"{'overall':14s}{right:8d}{len(rows):7d}{right / len(rows):10.1%}")

    answerable = [r for r in rows if r["category"] != "unanswerable"]
    refused = sum(r["refused_by_threshold"] for r in answerable)
    print(f"\nAnswerable questions refused by the score threshold: {refused} of {len(answerable)}")
    print(f"Saved per-question results to {out}")

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print("\nWrong answers:")
        for r in wrong:
            print(f"  {r['id']}: expected {r['expected']}\n      got: {r['answer'][:200]}")
    return summary
