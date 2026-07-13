from __future__ import annotations
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING
from rag.models import (
    MinimalSource,
    MinimalSearchResults,
    UnansweredQuestion,
    StudentSearchResults
)

if TYPE_CHECKING:
    import bm25s


class BM25Indexer:
    """An indexer and search engine utilizing the BM25 algorithm.

    Attributes:
        indexed_sources (list[MinimalSource]): A parallel list mapping
            corpus indices back to their original metadata sources.
    """

    def __init__(self) -> None:
        """Initializes an empty BM25Indexer instance."""
        self.indexed_sources: list[MinimalSource] = []

    def _tokenize(self, text: str) -> list[str]:
        """Normalizes text to lowercase and extracts alphanumeric tokens.

        Args:
            text (str): The raw text string to tokenize.

        Returns:
            list[str]: A list of clean word tokens.
        """
        return re.findall(r'\w+', text.lower())

    def _extract_file_chunks(
        self, file_path: str, file_sources: list[MinimalSource]
    ) -> list[str]:
        """Reads a file once and slices out raw text for all its chunks.

        Args:
            file_path (str): The path to the file on disk.
            file_sources (list[MinimalSource]): The list of metadata chunks
                belonging to this file.

        Returns:
            list[str]: A list of raw string contents for each chunk.
        """
        content: str = Path(file_path).read_text("utf-8")

        extracted_texts: list[str] = []
        for source in file_sources:
            chunk_start: int = source.first_character_index
            chunk_end: int = source.last_character_index

            sliced_text = content[chunk_start:chunk_end]
            extracted_texts.append(sliced_text)

        return extracted_texts

    def build_corpus(self, sources: list[MinimalSource]) -> list[list[str]]:
        """Groups sources by file, extracts text, and builds a corpus.

        Note:
            As a critical side effect, this method completely resets and
            populates the `indexed_sources` attribute to maintain a 1:1
            mapping alignment with the returned corpus.

        Args:
            sources (list[MinimalSource]): A flat list of all discovered
                chunk sources.

        Returns:
            list[list[str]]: A tokenized corpus ready for the BM25.
        """
        self.indexed_sources = []
        sources_by_file: defaultdict[str, list[MinimalSource]] = (
            defaultdict(list)
        )

        for source in sources:
            sources_by_file[source.file_path].append(source)

        corpus: list[list[str]] = []

        for file_path, file_sources in sources_by_file.items():
            raw_chunk_strings: list[str] = self._extract_file_chunks(
                file_path, file_sources
            )

            for i, raw_string in enumerate(raw_chunk_strings):
                tokenized_document: list[str] = self._tokenize(raw_string)
                corpus.append(tokenized_document)

                self.indexed_sources.append(file_sources[i])

        return corpus

    def search(
        self, query_string: str, retriever: bm25s.BM25, limit: int = 5
    ) -> list[MinimalSource]:
        """Searches the BM25 index and returns top matching chunk metadata.

        Args:
            query_string (str): The raw search query from the user.
            retriever (bm25s.BM25): The trained BM25 retrieval engine.
            limit (int): The maximum number of results to return.
                Defaults to 5.

        Returns:
            list[MinimalSource]: The top matching metadata sources.
        """
        query_tokens = self._tokenize(query_string)

        if limit > len(self.indexed_sources):
            limit = len(self.indexed_sources)

        indices, _ = retriever.retrieve([query_tokens], k=limit)

        # bm25s returns a 2D array for batch queries
        # indices[0] gets the matches for our single query
        return [self.indexed_sources[i] for i in indices[0]]

    def search_to_model(
        self,
        question_id: str,
        query_string: str,
        retriever: bm25s.BM25,
        limit: int = 5
    ) -> MinimalSearchResults:
        """Searches the index and wraps results in a Pydantic model.

        Args:
            question_id (str): The unique identifier for the question.
            query_string (str): The raw search query string.
            retriever (bm25s.BM25): The trained BM25 retrieval engine.
            limit (int): The maximum number of results to return.
                Defaults to 5.

        Returns:
            MinimalSearchResults: The validated search results container.
        """
        matching_sources: list[MinimalSource] = self.search(
            query_string, retriever, limit=limit
        )

        return MinimalSearchResults(
            question_id=question_id,
            question=query_string,
            retrieved_sources=matching_sources
        )

    def batch_search(
        self,
        questions: list[UnansweredQuestion],
        retriever: bm25s.BM25,
        limit: int = 5
    ) -> StudentSearchResults:
        """Processes multiple queries into a StudentSearchResults container.

        Args:
            questions (list[UnansweredQuestion]): A list of incoming
                unanswered questions.
            retriever (bm25s.BM25): The trained BM25 retrieval engine.
            limit (int): The maximum number of results per question.
                Defaults to 5.

        Returns:
            StudentSearchResults: The compiled batch results for the
                student platform.
        """
        results: list[MinimalSearchResults] = []

        for question in questions:
            search_result = self.search_to_model(
                question_id=question.question_id,
                query_string=question.question,
                retriever=retriever,
                limit=limit
            )
            results.append(search_result)

        return StudentSearchResults(search_results=results, k=limit)

    def save(self, save_dir: str, retriever: bm25s.BM25) -> None:
        """Saves BM25 index matrix and custom source metadata to disk.

        Args:
            save_dir (str): The directory path where index and metadata
                will be stored.
            retriever (bm25s.BM25): The BM25 retrieval engine instance
                to save.
        """
        retriever.save(save_dir)

        metadata_path: Path = Path(save_dir) / "metadata.json"
        serialized_sources = [
            source.model_dump() for source in self.indexed_sources
        ]

        metadata_path.write_text(
            json.dumps(serialized_sources, indent=4), encoding="utf-8"
        )

    def load_metadata(self, save_dir: str) -> None:
        """Loads custom source metadata from a JSON file.

        Args:
            save_dir (str): The directory path where the metadata file is
                located.
        """
        metadata_path: Path = Path(save_dir) / "metadata.json"
        raw_json: str = metadata_path.read_text(encoding="utf-8")
        self.indexed_sources = [
            MinimalSource(**item) for item in json.loads(raw_json)
        ]
