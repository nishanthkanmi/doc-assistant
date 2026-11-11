"""The four swappable parts of the pipeline, as simple interfaces.

To write your own, subclass one of these, fill in the methods, and put
"your_module:YourClass" as the `type` of that section in config.yaml.
Every component is created as  YourClass(options, config)  where `options`
is its own section of config.yaml (a dict) and `config` is the whole Config.

Data passed between parts:
  - a document is plain text plus a name (the source file's name without extension)
  - a chunk is a dict with at least
        "id"       unique, e.g. "AAPL-0042"
        "text"     the words to search and to show the language model
        "heading"  a one-line label saying where it came from
    and may carry any extra fields (section, company, ...) that get saved with it
  - a search result is a chunk dict plus "score" (higher = better match; the
    abstention threshold in config.yaml is compared against it)
"""

from abc import ABC, abstractmethod


class PipelineError(Exception):
    """A problem the user can fix (missing file, Ollama not running, bad setting)."""


class Component(ABC):
    def __init__(self, options, config):
        self.options = options
        self.config = config


class Cleaner(Component):
    """Turns one source file into plain text."""

    extensions = ()  # file types this cleaner reads, e.g. (".html", ".htm")

    @abstractmethod
    def clean(self, raw_bytes):
        """Return the readable text of one file, given its raw bytes."""


class Chunker(Component):
    """Splits one document's plain text into chunks."""

    @abstractmethod
    def chunk(self, text, doc_name):
        """Return a list of chunk dicts for one document."""


class Retriever(Component):
    """Builds a searchable index of chunks and finds the best ones for a question."""

    @abstractmethod
    def build(self, chunks):
        """Index the chunks and save the index to config.index_dir."""

    @abstractmethod
    def load(self):
        """Load the saved index. Raise PipelineError if there isn't one."""

    @abstractmethod
    def search(self, question, top_k):
        """Return the top_k best chunks, best first, each with a "score" added."""


class LanguageModel(Component):
    """Writes text from a prompt."""

    @abstractmethod
    def generate(self, system_prompt, user_prompt):
        """Return the model's reply as a string."""
