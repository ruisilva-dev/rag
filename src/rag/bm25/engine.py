from __future__ import annotations
import json
import bm25s
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from rag.bm25.utils import tokenize
from rag.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion
)


@dataclass
class SubIndex:
    """Container pairing a BM25 retriever instance with source metadata.

    Attributes:
        retriever (bm25s.BM25): The trained BM25 retrieval engine.
        sources (list[MinimalSource]): Parallel list mapping corpus indices to
            metadata.
    """

    retriever: bm25s.BM25
    sources: list[MinimalSource]


class BM25SearchEngine:
    """A search engine utilizing the BM25 algorithm with domain sub-indexing.

    Attributes:
        indices (dict[str, SubIndex]): Dictionary mapping domain identifiers
            ('code' and 'docs') to their respective sub-index components.
    """

    def __init__(
        self,
        code_retriever: bm25s.BM25,
        docs_retriever: bm25s.BM25,
        code_sources: list[MinimalSource],
        docs_sources: list[MinimalSource]
    ) -> None:
        """Initializes search engine with code and documentation sub-indices.

        Args:
            code_retriever (bm25s.BM25): BM25 retriever trained on Python code.
            docs_retriever (bm25s.BM25): BM25 retriever trained on Markdown
                docs.
            code_sources (list[MinimalSource]): Metadata sources parallel to
                the code index.
            docs_sources (list[MinimalSource]): Metadata sources parallel to
                the docs index.
        """
        self.indices: dict[str, SubIndex] = {
            "code": SubIndex(code_retriever, code_sources),
            "docs": SubIndex(docs_retriever, docs_sources)
        }

    @staticmethod
    def _load_sub_index(
        sub_dir: Path
    ) -> tuple[bm25s.BM25, list[MinimalSource]]:
        """Loads a single sub-index retriever and metadata from disk.

        Args:
            sub_dir (Path): Directory containing sub-index metadata and index
                files.

        Returns:
            tuple[bm25s.BM25, list[MinimalSource]]: Loaded retriever and source
                metadata.
        """
        metadata_path: Path = sub_dir / "metadata.json"
        raw_json: str = metadata_path.read_text(encoding="utf-8")
        indexed_sources = [
            MinimalSource(**item) for item in json.loads(raw_json)
        ]

        retriever = bm25s.BM25.load(sub_dir, load_corpus=False)

        return retriever, indexed_sources

    @staticmethod
    def _search_sub_index(
        query_string: str,
        retriever: bm25s.BM25,
        indexed_sources: list[MinimalSource],
        limit: int,
        is_code: bool
    ) -> tuple[list[MinimalSource], float]:
        """Retrieves top matches and top score from a specific sub-index.

        Args:
            query_string (str): The raw search query string.
            retriever (bm25s.BM25): BM25 retriever instance to query.
            indexed_sources (list[MinimalSource]): Source metadata parallel to
                index.
            limit (int): Maximum number of top hits to retrieve.
            is_code (bool): Whether query tokenization should use code mode.

        Returns:
            tuple[list[MinimalSource], float]: Top matching sources and top
                score.
        """
        query_tokens = tokenize(query_string, is_code)

        if limit > len(indexed_sources):
            limit = len(indexed_sources)

        indices, scores = retriever.retrieve([query_tokens], k=limit)

        top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0

        # bm25s returns a 2D array for batch queries
        # indices[0] gets the matches for our single query
        return [indexed_sources[i] for i in indices[0]], top_score

    @staticmethod
    def _detect_query_intent(query_string: str) -> str | None:
        """Detects if a query targets code or docs via pattern matching.

        Args:
            query_string (str): The raw user search query string.

        Returns:
            str | None: Intent string ('docs' or 'code'), or None if ambiguous.
        """
        query_lower = query_string.lower()

        # Documentation signals
        docs_patterns = [
            "how to", "how do", "where can i", "where is",
            "installation", "quickstart", "setup instructions",
            "tutorial"
        ]
        if any(pattern in query_lower for pattern in docs_patterns):
            return "docs"

        # Code signals
        code_patterns = ["class ", "def ", "return ", "assert "]
        has_code_syntax = (
            bool(re.search(r"\w\(", query_string)) or  # Example: process(path)
            "_" in query_string or
            ".py" in query_lower
        )

        if (
            any(pattern in query_lower for pattern in code_patterns) or
            has_code_syntax
        ):
            return "code"

        # Ambiguous query fallback
        return None

    @staticmethod
    def _rrf_fuse(
        code_sources: list[MinimalSource],
        docs_sources: list[MinimalSource],
        limit: int
    ) -> list[MinimalSource]:
        """Fuses code and docs result lists using Reciprocal Rank Fusion (RRF).

        Args:
            code_sources (list[MinimalSource]): Ranked sources from code index.
            docs_sources (list[MinimalSource]): Ranked sources from docs index.
            limit (int): Maximum number of fused sources to return.

        Returns:
            list[MinimalSource]: Combined and re-ranked list of top sources.
        """
        rrf_scores: defaultdict[tuple[str, int, int], float] = defaultdict(
            float
        )
        source_map: dict[tuple[str, int, int], MinimalSource] = {}

        # limit as the smoothing constant k in the RRF equation
        k_constant = limit

        for source_list in (code_sources, docs_sources):
            for rank, source in enumerate(source_list, start=1):
                # Key for source, pydantic models aren't naturally hashable
                chunk_key = (
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index
                )

                # Accumulate score using RRF equation
                rrf_scores[chunk_key] += 1.0 / (k_constant + rank)
                source_map[chunk_key] = source

        # Sort sources by RRF score descending
        sorted_keys = sorted(
            rrf_scores.keys(),
            key=lambda key: rrf_scores[key],
            reverse=True
        )

        return [source_map[key] for key in sorted_keys[:limit]]

    @classmethod
    def load_from_disk(cls, save_dir: str) -> BM25SearchEngine:
        """Loads the BM25 index matrices and source metadata from disk.

        Args:
            save_dir (str): Directory path containing saved sub-index folders.

        Returns:
            BM25SearchEngine: A fully initialized search engine.

        Raises:
            FileNotFoundError: If metadata file or index folder is missing.
        """
        code_path: Path = Path(save_dir) / "code"
        docs_path: Path = Path(save_dir) / "docs"

        code_retriever, code_sources = cls._load_sub_index(code_path)
        docs_retriever, docs_sources = cls._load_sub_index(docs_path)

        return cls(code_retriever, docs_retriever, code_sources, docs_sources)

    def search(
        self, query_string: str, limit: int = 10
    ) -> list[MinimalSource]:
        """Searches the BM25 indices and returns top matching chunk metadata.

        Args:
            query_string (str): The raw search query from the user.
            limit (int): Maximum number of results to return. Defaults to 5.

        Returns:
            list[MinimalSource]: The top matching metadata sources.
        """
        oversample_limit = limit * 10

        code_sources, code_top_score = self._search_sub_index(
            query_string,
            self.indices["code"].retriever,
            self.indices["code"].sources,
            oversample_limit,
            is_code=True
        )

        docs_sources, docs_top_score = self._search_sub_index(
            query_string,
            self.indices["docs"].retriever,
            self.indices["docs"].sources,
            oversample_limit,
            is_code=False
        )

        if code_top_score == 0.0:
            return docs_sources[:limit]
        if docs_top_score == 0.0:
            return code_sources[:limit]

        query_intent = self._detect_query_intent(query_string)

        normalized_docs_score = docs_top_score * 2.0
        score_ratio = code_top_score / normalized_docs_score

        if query_intent == "code" and code_top_score >= normalized_docs_score:
            return code_sources[:limit]
        elif query_intent == "docs" and normalized_docs_score > code_top_score:
            return docs_sources[:limit]

        if score_ratio > 1.25:
            return code_sources[:limit]
        elif score_ratio < 0.8:
            return docs_sources[:limit]
        else:
            return self._rrf_fuse(code_sources, docs_sources, limit)

    def search_to_model(
        self,
        question_id: str,
        query_string: str,
        limit: int = 5
    ) -> MinimalSearchResults:
        """Searches the index and wraps results in a Pydantic model.

        Args:
            question_id (str): The unique identifier for the question.
            query_string (str): The raw search query string.
            limit (int): Maximum number of results to return. Defaults to 5.

        Returns:
            MinimalSearchResults: Validated search results container.
        """
        matching_sources: list[MinimalSource] = self.search(
            query_string, limit=limit
        )

        return MinimalSearchResults(
            question_id=question_id,
            question=query_string,
            retrieved_sources=matching_sources
        )

    def batch_search(
        self,
        questions: list[UnansweredQuestion],
        limit: int = 5
    ) -> StudentSearchResults:
        """Processes multiple queries into a StudentSearchResults container.

        Args:
            questions (list[UnansweredQuestion]): List of incoming unanswered
                questions.
            limit (int): Maximum number of results per question. Defaults to 5.

        Returns:
            StudentSearchResults: The compiled batch results.
        """
        results: list[MinimalSearchResults] = []

        for question in questions:
            search_result = self.search_to_model(
                question_id=question.question_id,
                query_string=question.question,
                limit=limit
            )
            results.append(search_result)

        return StudentSearchResults(search_results=results, k=limit)
