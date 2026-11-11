"""Loads config.yaml, the single place where every setting lives.

    config = load_config()               # reads config.yaml in the project folder
    config = load_config("other.yaml")   # or any other file
    config.section("chunker")["max_chars"]
    config.clean_dir                     # derived folders, from documents.workspace
"""

from pathlib import Path

from ruamel.yaml import YAML

from interfaces import PipelineError

ROOT = Path(__file__).resolve().parent.parent  # the project folder, wherever you run from

REQUIRED_SECTIONS = [
    "documents", "cleaner", "chunker", "retriever", "language_model",
    "abstention", "answering", "evaluation",
]


class Config:
    def __init__(self, data, path):
        self.data = data
        self.path = Path(path)
        self.root = self.path.parent  # paths in the file are relative to this folder
        for name in REQUIRED_SECTIONS:
            if name not in data or not isinstance(data[name], dict):
                raise PipelineError(f"{self.path.name} is missing the '{name}' section.")

    def section(self, name):
        """The settings under one top-level key, as a dict."""
        if name not in self.data:
            raise PipelineError(f"{self.path.name} has no '{name}' section.")
        return self.data[name]

    def _path(self, relative):
        return (self.root / relative).resolve()

    @property
    def source_dir(self):
        return self._path(self.section("documents")["folder"])

    @property
    def workspace(self):
        return self._path(self.section("documents")["workspace"])

    @property
    def clean_dir(self):
        return self.workspace / "clean"

    @property
    def chunks_file(self):
        return self.workspace / "chunks" / "chunks.jsonl"

    @property
    def index_dir(self):
        return self.workspace / "index"

    @property
    def questions_file(self):
        return self._path(self.section("evaluation")["questions_file"])

    @property
    def results_file(self):
        return self.workspace / "eval" / "results.csv"


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.exists():
        raise PipelineError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = YAML(typ="safe").load(f)
    return Config(data, path)
