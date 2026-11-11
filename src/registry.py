"""Turns the `type` setting of each config.yaml section into a working component.

A type is either a built-in name from BUILTIN below, or "module.path:ClassName"
for your own class (the module must be importable from the project or src/ folder).
"""

import importlib

from interfaces import Chunker, Cleaner, LanguageModel, PipelineError, Retriever

BUILTIN = {
    "cleaner": {
        "html": "components.cleaners:HtmlCleaner",
        "text": "components.cleaners:TextCleaner",
    },
    "chunker": {
        "sec_filing": "components.chunkers:SecFilingChunker",
        "paragraph": "components.chunkers:ParagraphChunker",
    },
    "retriever": {
        "ollama_embeddings": "components.retrievers:OllamaEmbeddingRetriever",
        "tfidf": "components.retrievers:TfidfRetriever",
    },
    "language_model": {
        "ollama": "components.language_models:OllamaChat",
    },
}

INTERFACES = {
    "cleaner": Cleaner,
    "chunker": Chunker,
    "retriever": Retriever,
    "language_model": LanguageModel,
}


def build_component(kind, config):
    """Create the component named by config.yaml's `kind` section."""
    options = config.section(kind)
    name = options.get("type")
    if not name:
        raise PipelineError(f"The '{kind}' section of {config.path.name} needs a 'type'.")

    target = BUILTIN[kind].get(name, name)
    if ":" not in target:
        known = ", ".join(BUILTIN[kind])
        raise PipelineError(
            f"Unknown {kind} type '{name}'. Built-in types: {known}. "
            f"For your own, write 'module.path:ClassName'."
        )
    module_name, _, class_name = target.partition(":")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as error:
        raise PipelineError(f"Could not load {kind} '{target}': {error}")
    if not (isinstance(cls, type) and issubclass(cls, INTERFACES[kind])):
        raise PipelineError(f"{target} must be a subclass of interfaces.{INTERFACES[kind].__name__}.")
    return cls(options, config)


def build_components(config):
    """Create all four components. Returns a dict keyed by kind."""
    return {kind: build_component(kind, config) for kind in INTERFACES}
