"""The pipeline stages. Each one takes the config plus the component it needs,
so any component can be swapped in config.yaml without touching this file.

    clean     source documents  ->  workspace/clean/*.txt          (Cleaner)
    chunk     cleaned text      ->  workspace/chunks/chunks.jsonl   (Chunker)
    index     chunks            ->  workspace/index/                (Retriever)
    ask       question          ->  answer + the chunks it used     (Retriever + LanguageModel)
"""

import json

from interfaces import PipelineError


def source_files(config, extensions=None):
    folder = config.source_dir
    if not folder.is_dir():
        raise PipelineError(f"The documents folder {folder} does not exist. Check documents.folder in config.yaml.")
    files = sorted(p for p in folder.iterdir() if p.is_file())
    if extensions is not None:
        files = [p for p in files if p.suffix.lower() in extensions]
    return files


def run_clean(config, cleaner):
    paths = source_files(config, cleaner.extensions)
    if not paths:
        raise PipelineError(
            f"No {'/'.join(cleaner.extensions)} files in {config.source_dir}. "
            "Put your documents there, or pick a different cleaner type in config.yaml."
        )
    config.clean_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        print(f"{path.name}: cleaning...")
        text = cleaner.clean(path.read_bytes())
        out_path = config.clean_dir / f"{path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        before_mb = path.stat().st_size / 1_000_000
        after_mb = out_path.stat().st_size / 1_000_000
        print(f"{path.name}: saved {out_path.name} ({before_mb:.1f} MB -> {after_mb:.1f} MB)")


def run_chunk(config, chunker):
    # Only chunk files that came from the current documents folder, so leftovers
    # from an earlier folder in the same workspace are ignored.
    stems = {p.stem for p in source_files(config)}
    paths = sorted(p for p in config.clean_dir.glob("*.txt") if p.stem in stems)
    if not paths:
        raise PipelineError(f"No cleaned text in {config.clean_dir}. Run: py run_all.py --stages clean")

    all_chunks = []
    for path in paths:
        chunks = chunker.chunk(path.read_text(encoding="utf-8"), path.stem)
        all_chunks.extend(chunks)
        sizes = [len(c["text"]) for c in chunks]
        print(f"{path.stem}: {len(chunks)} chunks (average {sum(sizes) // len(sizes)} characters, largest {max(sizes)})")

    config.chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.chunks_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {config.chunks_file}")


def run_index(config, retriever):
    if not config.chunks_file.exists():
        raise PipelineError(f"{config.chunks_file} not found. Run: py run_all.py --stages chunk")
    with open(config.chunks_file, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    retriever.build(chunks)


def build_prompt(question, chunks):
    """Put the question and the numbered source chunks into one message."""
    sources = [f"[{number}] {chunk['heading']}\n{chunk['text']}" for number, chunk in enumerate(chunks, start=1)]
    return "SOURCES\n\n" + "\n\n---\n\n".join(sources) + f"\n\nQUESTION: {question}"


def ask_question(question, retriever, language_model, config):
    """Retrieve chunks and answer. Returns a dict with the answer and its evidence.

    If the best chunk scores below abstention.min_score, the language model is
    not called at all and the answer is the abstention message.
    """
    top_k = config.section("retriever")["top_k"]
    found = retriever.search(question, top_k)
    top_score = found[0]["score"]
    abstained = top_score < config.section("abstention")["min_score"]
    if abstained:
        reply = config.section("abstention")["message"]
    else:
        reply = language_model.generate(config.section("answering")["system_prompt"], build_prompt(question, found))
    return {"answer": reply, "sources": found, "top_score": top_score, "abstained": abstained}
