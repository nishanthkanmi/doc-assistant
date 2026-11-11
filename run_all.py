"""Runs the whole pipeline on the document folder named in config.yaml.

    py run_all.py                        # clean, chunk, index, evaluate
    py run_all.py --stages chunk,index   # only some stages, in pipeline order
    py run_all.py --config other.yaml    # use a different config file

Stages:
    clean     source documents -> plain text            (cleaner)
    chunk     plain text       -> chunks                (chunker)
    index     chunks           -> search index          (retriever)
    evaluate  questions file   -> accuracy per category (retriever + language model)

Ollama must be running for the ollama components. To ask a single question
afterwards, use: py src/ask.py "your question"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # so the modules in src/ can be imported

from config import load_config  # noqa: E402
from evaluation import run_evaluation  # noqa: E402
from interfaces import PipelineError  # noqa: E402
from pipeline import run_chunk, run_clean, run_index  # noqa: E402
from registry import build_components  # noqa: E402

STAGES = ["clean", "chunk", "index", "evaluate"]


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Run the document assistant pipeline")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    parser.add_argument("--stages", default=",".join(STAGES),
                        help=f"comma-separated stages to run (default: all of {','.join(STAGES)})")
    args = parser.parse_args()

    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. Choose from: {', '.join(STAGES)}")

    try:
        config = load_config(args.config)
        parts = build_components(config)

        print(f"Config:    {config.path}")
        print(f"Documents: {config.source_dir}")
        print(f"Workspace: {config.workspace}")
        for kind, part in parts.items():
            print(f"{kind + ':':16s}{config.section(kind)['type']}  ({type(part).__name__})")

        if "clean" in wanted:
            banner("Stage 1/4: clean")
            run_clean(config, parts["cleaner"])
        if "chunk" in wanted:
            banner("Stage 2/4: chunk")
            run_chunk(config, parts["chunker"])
        if "index" in wanted:
            banner("Stage 3/4: index")
            run_index(config, parts["retriever"])
        if "evaluate" in wanted:
            banner("Stage 4/4: evaluate")
            if not config.questions_file.exists():
                print(f"Skipped: no questions file at {config.questions_file}.\n"
                      "Write one (format: see data/eval/questions.json) or point "
                      "evaluation.questions_file in config.yaml at it.")
            else:
                parts["retriever"].load()  # load from disk, exactly as ask.py does
                run_evaluation(config, parts["retriever"], parts["language_model"])
    except PipelineError as error:
        raise SystemExit(f"\nError: {error}")


if __name__ == "__main__":
    main()
