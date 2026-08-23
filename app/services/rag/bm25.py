import math
import re
from typing import Any

from app.services.document.models import EmbeddedChunk, MetadataFilter, normalize_metadata_filter


def tokenize(text: str) -> list[str]:
    """Tokenize input text into lowercase alphanumeric word tokens."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


class BM25Index:
    """In-memory Okapi BM25 index and scoring engine for lexical chunk retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[EmbeddedChunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._num_docs: int = 0

    @property
    def corpus_size(self) -> int:
        return self._num_docs

    def build_index(
        self,
        chunks: list[EmbeddedChunk],
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> None:
        """Build term frequency and inverse document frequency index from candidate chunks."""
        filter_obj = normalize_metadata_filter(filters)
        filtered_chunks = [
            c for c in chunks if filter_obj is None or filter_obj.matches(c.metadata)
        ]

        self._chunks = filtered_chunks
        self._num_docs = len(filtered_chunks)
        self._doc_tokens = []
        self._doc_lens = []
        self._doc_freqs = {}
        self._idf = {}

        if self._num_docs == 0:
            self._avgdl = 0.0
            return

        total_length = 0
        for chunk in filtered_chunks:
            tokens = tokenize(chunk.content)
            self._doc_tokens.append(tokens)
            doc_len = len(tokens)
            self._doc_lens.append(doc_len)
            total_length += doc_len

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        self._avgdl = total_length / self._num_docs

        # Calculate standard smoothed IDF for each observed term:
        # IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
        for term, freq in self._doc_freqs.items():
            numerator = self._num_docs - freq + 0.5
            denominator = freq + 0.5
            self._idf[term] = math.log((numerator / denominator) + 1.0)

    def score(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Score candidate chunks against query terms using the Okapi BM25 formula."""
        if self._num_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[EmbeddedChunk, float]] = []

        for idx, (chunk, doc_tokens, doc_len) in enumerate(
            zip(self._chunks, self._doc_tokens, self._doc_lens, strict=False)
        ):
            # Compute term frequencies for doc
            tf_dict: dict[str, int] = {}
            for t in doc_tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            doc_score = 0.0
            for term in query_tokens:
                if term not in tf_dict:
                    continue

                tf = tf_dict[term]
                idf = self._idf.get(term, 0.0)

                # Okapi BM25 term weighting component
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self._avgdl or 1.0)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / denom)
                doc_score += term_score

            if doc_score >= min_score and doc_score > 0.0:
                scores.append((chunk, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
