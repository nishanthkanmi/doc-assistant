"""Retrievers: build a searchable index of chunks and find the best ones for a question.

Both retrievers save their index in config.index_dir, and both put each chunk's
heading in front of its text before indexing, so a table of numbers still
"knows" which company and section it belongs to.
"""

import json
import math
import re
from collections import Counter

import numpy as np

from components.ollama import ollama_post
from interfaces import PipelineError, Retriever


def indexed_text(chunk):
    return chunk["heading"] + "\n\n" + chunk["text"]


class OllamaEmbeddingRetriever(Retriever):
    """Finds chunks by meaning, using a free local embedding model served by Ollama.

    An embedding is a list of numbers that captures what a text means; texts with
    similar meanings get similar numbers. The score is the cosine similarity
    between the question and a chunk (1.0 = identical meaning).

    Files written: embeddings.npy (one row per chunk), chunks.json, info.json.
    """

    def __init__(self, options, config):
        super().__init__(options, config)
        self.model = options["embedding_model"]
        self.chunks = None
        self.embeddings = None

    def _embed(self, texts, prefix):
        """Return a numpy array with one unit-length embedding per text."""
        reply = ollama_post(self.config, "/api/embed", {"model": self.model, "input": [prefix + t for t in texts]})
        vectors = np.array(reply["embeddings"], dtype=np.float32)
        # Scale each vector to length 1, so comparing two of them is a simple dot product.
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def build(self, chunks):
        batch_size = self.options.get("batch_size", 32)
        texts = [indexed_text(c) for c in chunks]
        batches = []
        for start in range(0, len(texts), batch_size):
            batches.append(self._embed(texts[start:start + batch_size], self.options.get("document_prefix", "")))
            print(f"Embedded {min(start + batch_size, len(texts))} of {len(texts)} chunks", end="\r")
        print()
        embeddings = np.vstack(batches)  # stack the batches into one big table

        index_dir = self.config.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", embeddings)
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        with open(index_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump({"embedding_model": self.model, "chunks": len(chunks), "dimensions": embeddings.shape[1]}, f)
        print(f"Saved index of {len(chunks)} chunks ({embeddings.shape[1]} numbers each) to {index_dir}")
        self.chunks, self.embeddings = chunks, embeddings

    def load(self):
        index_dir = self.config.index_dir
        if not (index_dir / "embeddings.npy").exists():
            raise PipelineError(f"No index in {index_dir}. Run: py run_all.py --stages index")
        with open(index_dir / "info.json", encoding="utf-8") as f:
            info = json.load(f)
        if info["embedding_model"] != self.model:
            raise PipelineError(
                f"The index was built with {info['embedding_model']!r} but config.yaml says "
                f"{self.model!r}. Rebuild it with: py run_all.py --stages index"
            )
        self.embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, question, top_k):
        query = self._embed([question], self.options.get("query_prefix", ""))[0]
        scores = self.embeddings @ query  # dot product with every chunk at once
        best = np.argsort(scores)[::-1][:top_k]  # positions of the highest scores
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfRetriever(Retriever):
    """Finds chunks by shared words, with no model needed.

    Each word is weighted by how rare it is across all chunks (TF-IDF), and the
    score is the cosine similarity of the word weights. These scores run much
    lower than embedding scores (a good match is often 0.2 to 0.4), so give
    abstention.min_score in config.yaml a value to match.

    Files written: tfidf.json.
    """

    def __init__(self, options, config):
        super().__init__(options, config)
        self.chunks = None
        self.idf = None
        self.vectors = None

    def _vector(self, text):
        """Unit-length {word: weight} for a text, using the stored word rarities."""
        weights = {w: (1 + math.log(n)) * self.idf[w] for w, n in Counter(tokens(text)).items() if w in self.idf}
        norm = math.sqrt(sum(x * x for x in weights.values())) or 1.0
        return {w: x / norm for w, x in weights.items()}

    def build(self, chunks):
        documents = [set(tokens(indexed_text(c))) for c in chunks]
        frequency = Counter(word for words in documents for word in words)
        n = len(chunks)
        self.idf = {w: math.log((n + 1) / (df + 1)) + 1 for w, df in frequency.items()}
        self.chunks = chunks
        self.vectors = [self._vector(indexed_text(c)) for c in chunks]

        index_dir = self.config.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "tfidf.json", "w", encoding="utf-8") as f:
            json.dump({"idf": self.idf, "vectors": self.vectors, "chunks": chunks}, f, ensure_ascii=False)
        print(f"Saved word index of {len(chunks)} chunks ({len(self.idf)} distinct words) to {index_dir}")

    def load(self):
        path = self.config.index_dir / "tfidf.json"
        if not path.exists():
            raise PipelineError(f"No index at {path}. Run: py run_all.py --stages index")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.idf, self.vectors, self.chunks = data["idf"], data["vectors"], data["chunks"]

    def search(self, question, top_k):
        query = self._vector(question)
        scores = [sum(weight * vector.get(word, 0.0) for word, weight in query.items()) for vector in self.vectors]
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [{**self.chunks[i], "score": scores[i]} for i in best]
