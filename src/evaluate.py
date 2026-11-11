"""Run only the evaluation stage (for the whole pipeline, use run_all.py).

    py src/evaluate.py
    py src/evaluate.py --category numeric     # one category, for quick checks
    py src/evaluate.py --config other.yaml

Ollama must be running, and the index must already exist.
"""

import argparse

from config import load_config
from evaluation import CATEGORIES, run_evaluation
from interfaces import PipelineError
from registry import build_component


def main():
    parser = argparse.ArgumentParser(description="Score the pipeline on the questions file")
    parser.add_argument("--category", choices=CATEGORIES, help="only run this category")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        retriever = build_component("retriever", config)
        retriever.load()
        run_evaluation(config, retriever, build_component("language_model", config), args.category)
    except PipelineError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
