"""Ask a question about the documents and get an answer with its sources.

    py src/ask.py "What were Apple's total net sales in 2025?"
    py src/ask.py --config other.yaml "your question"

The index must already exist (py run_all.py builds it). evaluate.py uses the
same ask_question() function, so the test measures exactly this pipeline.
"""

import argparse

from config import load_config
from interfaces import PipelineError
from pipeline import ask_question
from registry import build_component


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the documents")
    parser.add_argument("question", nargs="+", help="your question")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    args = parser.parse_args()
    question = " ".join(args.question)

    try:
        config = load_config(args.config)
        retriever = build_component("retriever", config)
        retriever.load()
        language_model = build_component("language_model", config)
        result = ask_question(question, retriever, language_model, config)
    except PipelineError as error:
        raise SystemExit(str(error))

    lm = config.section("language_model")
    min_score = config.section("abstention")["min_score"]
    print(f"\nQuestion: {question}\n")
    if result["abstained"]:
        print(f"Answer (no model call: best score {result['top_score']:.3f} is below min_score {min_score}):")
    else:
        print(f"Answer ({lm.get('name', lm['type'])}):")
    print(f"{result['answer']}\n")
    print("=" * 70)
    print(f"Sources (top {len(result['sources'])} chunks):")
    for number, chunk in enumerate(result["sources"], start=1):
        print(f"\n[{number}] {chunk['id']}  score {chunk['score']:.3f}")
        print(f"    {chunk['heading']}")
        print("    " + chunk["text"].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
