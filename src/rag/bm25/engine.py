from __future__ import annotations
import json
import bm25s
from pathlib import Path
from rag.bm25.utils import tokenize
from rag.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion
)


class BM25SearchEngine:
    """A search engine utilizing the BM25 algorithm for ranked text retrieval.

    Attributes:
        retriever (bm25s.BM25): The underlying loaded BM25 scoring model.
        indexed_sources (list[MinimalSource]): A parallel list mapping
            corpus indices back to their original metadata sources.
    """

    def __init__(
        self, retriever: bm25s.BM25, indexed_sources: list[MinimalSource]
    ) -> None:
        """Initializes the search engine with a retriever and source metadata.

        Args:
            retriever (bm25s.BM25): The trained BM25 retrieval engine instance.
            indexed_sources (list[MinimalSource]): A parallel list mapping
                corpus indices back to their original metadata sources.
        """
        self.retriever: bm25s.BM25 = retriever
        self.indexed_sources: list[MinimalSource] = indexed_sources

    @classmethod
    def load_from_disk(cls, save_dir: str) -> BM25SearchEngine:
        """Loads the BM25 index matrices and source metadata from disk.

        Args:
            save_dir (str): The directory path containing the saved index
                and 'metadata.json'.

        Returns:
            BM25SearchEngine: A fully initialized search engine.

        Raises:
            FileNotFoundError: If the metadata file or index directory
                does not exist.
        """
        metadata_path: Path = Path(save_dir) / "metadata.json"
        raw_json: str = metadata_path.read_text(encoding="utf-8")
        indexed_sources = [
            MinimalSource(**item) for item in json.loads(raw_json)
        ]

        retriever = bm25s.BM25.load(save_dir, load_corpus=False)

        return cls(retriever, indexed_sources)

    def search(
        self, query_string: str, limit: int = 5
    ) -> list[MinimalSource]:
        """Searches the BM25 index and returns top matching chunk metadata.

        Args:
            query_string (str): The raw search query from the user.
            limit (int): The maximum number of results to return.
                Defaults to 5.

        Returns:
            list[MinimalSource]: The top matching metadata sources.
        """
        query_tokens = tokenize(query_string)

        if limit > len(self.indexed_sources):
            limit = len(self.indexed_sources)

        indices, _ = self.retriever.retrieve([query_tokens], k=limit)

        # bm25s returns a 2D array for batch queries
        # indices[0] gets the matches for our single query
        return [self.indexed_sources[i] for i in indices[0]]

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
            limit (int): The maximum number of results to return.
                Defaults to 5.

        Returns:
            MinimalSearchResults: The validated search results container.
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
            questions (list[UnansweredQuestion]): A list of incoming
                unanswered questions.
            limit (int): The maximum number of results per question.
                Defaults to 5.

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
