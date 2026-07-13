from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING
from rag.models import MinimalSource
from rag.bm25.utils import tokenize

if TYPE_CHECKING:
    import bm25s


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
                tokenized_document: list[str] = tokenize(raw_string)
                corpus.append(tokenized_document)

                self.indexed_sources.append(file_sources[i])

        return corpus

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
