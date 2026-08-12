"""BM25 search engine for querying indexed document corpora."""

from __future__ import annotations
import json
import bm25s
import Stemmer
from pathlib import Path
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion
)


class BM25SearchEngine:
    """A search engine utilizing the BM25 algorithm.

    Attributes:
        sources (list[MinimalSource]): A parallel list mapping
            corpus indices back to their original metadata sources.
    """

    def __init__(
        self, retriever: bm25s.BM25, sources: list[MinimalSource]
    ) -> None:
        """Initializes search engine with index and source metadata.

        Args:
            retriever (bm25s.BM25): Loaded BM25 search index.
            sources (list[MinimalSource]): Parallel list mapping index
                positions back to original metadata sources.
        """
        self.retriever: bm25s.BM25 = retriever
        self.sources: list[MinimalSource] = sources

    @classmethod
    def load_from_disk(cls, save_dir: str) -> BM25SearchEngine:
        """Loads the BM25 index and source metadata from disk.

        Args:
            save_dir (str): Directory path containing saved index.

        Returns:
            BM25SearchEngine: A fully initialized search engine.

        Raises:
            FileNotFoundError: If metadata file or index folder is missing.
        """
        metadata_path: Path = Path(save_dir) / "metadata.json"
        raw_json: str = metadata_path.read_text(encoding="utf-8")
        sources = [
            MinimalSource(**item) for item in json.loads(raw_json)
        ]

        retriever = bm25s.BM25.load(save_dir, load_corpus=False)

        return cls(retriever=retriever, sources=sources)

    def search(
        self, query_string: str, limit: int = 10
    ) -> list[MinimalSource]:
        """Searches the BM25 index and returns top matching chunk metadata.

        Args:
            query_string (str): The raw search query from the user.
            limit (int): Maximum number of results to return. Defaults to 10.

        Returns:
            list[MinimalSource]: The top matching metadata sources.
        """
        expanded_limit = limit * 10

        stemmer = Stemmer.Stemmer("english")
        stop_words: list[str] = list(bm25s.stopwords.STOPWORDS_EN_PLUS)

        query_tokens = bm25s.tokenize(
            query_string, stemmer=stemmer, stopwords=stop_words
        )

        if limit > len(self.sources):
            limit = len(self.sources)

        if expanded_limit > len(self.sources):
            expanded_limit = limit

        indices, _ = self.retriever.retrieve(query_tokens, k=expanded_limit)

        # bm25s returns a 2D array for batch queries
        # indices[0] gets the matches for our single query
        results: list[MinimalSource] = [self.sources[i] for i in indices[0]]

        return results[:limit]

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
            question_str=query_string,
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
