import pydantic
import sys
from pathlib import Path
from rag.ingestion.base import BaseParser
from rag.ingestion.py_parser import PythonParser
from rag.ingestion.md_parser import MarkdownParser
from rag.models import MinimalSource

Chunk = tuple[str, int, int, list[str]]


class IngestionPipeline:
    """A text-splitting engine for Python code and Markdown files.

    Attributes:
        max_chunk_size (int): The maximum character length permitted
            for an individual text chunk.
        parsers (dict[str, BaseParser, list[Chunk]]): A dispatch mapping of
            supported file extensions to their respective processing objects.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes the FileChunker with a maximum chunk size.

        Args:
            max_chunk_size (int): Maximum characters per chunk.
                Defaults to 2000.
        """
        self.max_chunk_size: int = max_chunk_size
        self.parsers: dict[str, BaseParser] = {
            ".py": PythonParser(max_chunk_size),
            ".md": MarkdownParser(max_chunk_size)
        }

    def process_file(self, file_path: Path) -> list[MinimalSource]:
        """Reads a file and delegates it to the correct parser.

        Errors during file reading or encoding are caught and logged to
        standard error, returning an empty list to ensure the larger pipeline
        does not crash.

        Args:
            file_path (Path): The path to the file to be processed.

        Returns:
            list[MinimalSource]: A list of validated chunk metadata sources.
                Returns an empty list if an error occurs.
        """
        parser = self.parsers.get(file_path.suffix)
        if parser is None:
            return []

        raw_chunks = parser.parse(file_path)

        # Package the raw chunks into Pydantic models
        sources = []
        for _, start, end, headers in raw_chunks:
            try:
                sources.append(
                    MinimalSource(
                        file_path=str(file_path),
                        first_character_index=start,
                        last_character_index=end,
                        context_headers=headers
                    )
                )
            except pydantic.ValidationError as e:
                print(
                    "Error: Invalid data structure in input file.",
                    file=sys.stderr
                )
                for error in e.errors():
                    field = " -> ".join(str(loc) for loc in error["loc"])
                    msg = error["msg"]
                    print(f"  - Field '{field}': {msg}", file=sys.stderr)
        return sources
