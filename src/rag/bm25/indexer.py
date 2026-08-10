"""Corpus construction and BM25 indexing pipeline."""

from __future__ import annotations
import json
import bm25s
import Stemmer
from pathlib import Path
from collections import defaultdict
from rag.models import MinimalSource


class BM25Indexer:
    """An indexer utilizing the BM25 algorithm.

    Attributes:
        indexed_sources (list[MinimalSource]): A parallel list mapping
            corpus indices back to their original metadata sources.
    """

    def __init__(self) -> None:
        """Initializes an empty BM25Indexer instance."""
        self.indexed_sources: list[MinimalSource] = []

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

    def build_corpus(
        self, sources: list[MinimalSource]
    ) -> list[list[str]] | bm25s.tokenization.Tokenized:
        """Groups sources by file, extracts text, and builds a corpus.

        Note:
            As a critical side effect, this method completely resets and
            populates the `indexed_sources` attribute to maintain a 1:1
            mapping alignment with the returned corpus.

        Args:
            sources (list[MinimalSource]): A flat list of all discovered
                chunk sources.

        Returns:
            list[list[str]] | bm25s.tokenization.Tokenized: A tokenized corpus
                ready for the BM25.
        """
        self.indexed_sources = []
        sources_by_file: defaultdict[str, list[MinimalSource]] = (
            defaultdict(list)
        )

        for source in sources:
            sources_by_file[source.file_path].append(source)

        texts: list[str] = []
        stemmer = Stemmer.Stemmer("english")
        stop_words: list[str] = list(bm25s.stopwords.STOPWORDS_EN_PLUS)

        for file_path, file_sources in sources_by_file.items():
            raw_chunk_strings: list[str] = self._extract_file_chunks(
                file_path, file_sources
            )

            for i, raw_string in enumerate(raw_chunk_strings):
                header_trail = " > ".join(file_sources[i].context_headers)
                combined_string = f"{file_path}: {header_trail}\n{raw_string}"
                texts.append(combined_string)
                self.indexed_sources.append(file_sources[i])

        corpus_tokens = bm25s.tokenize(
            texts, stemmer=stemmer, stopwords=stop_words
        )

        return corpus_tokens

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
