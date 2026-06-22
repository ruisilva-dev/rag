from pathlib import Path
from typing import Generator
import itertools
import re
from rag.models import MinimalSource


class FileDiscoverer:
    """A discoverer for locating source and documentation files.

    Attributes:
        repo_path (Path): The root directory path where the file
            search begins.
    """

    def __init__(self, repo_path: str | Path) -> None:
        """Initializes the FileDiscoverer with a repository path.

        Args:
            repo_path (str | Path): The path to the target repository.
        """
        self.repo_path = Path(repo_path)

    def discover_files(self) -> Generator[Path, None, None]:
        """Recursively discovers all Python (.py) and Markdown (.md) files.

        Uses lazy generator chaining to stream files efficiently from the
        filesystem without loading all paths into memory at once.

        Yields:
            Path: The path object of a discovered file matching the
                target extensions.
        """
        py_files = self.repo_path.rglob("*.py")
        md_files = self.repo_path.rglob("*.md")

        for file_path in itertools.chain(py_files, md_files):
            yield file_path


class FileChunker:
    """A text-splitting engine for Python code and Markdown files.

    Attributes:
        max_chunk_size (int): The maximum character length permitted
            for an individual text chunk.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes the FileChunker with a maximum chunk size.

        Args:
            max_chunk_size (int): Maximum characters per chunk.
        """
        self.max_chunk_size = max_chunk_size

    def _process_py(self, file_path: Path) -> list[tuple[str, int, int]]:
        return []

    def _process_md(self, file_path: Path) -> list[tuple[str, int, int]]:
        content: str = file_path.read_text(encoding="utf-8")
        current_start: int = 0

        blocks: list[tuple[str, int, int]] = []

        # Slice content between matches (paragraphs)
        for match in re.finditer(r"\n\n", content):
            block_text: str = content[current_start:match.start()]
            blocks.append((block_text, current_start, match.start()))

            current_start = match.end()

        # Cleanup for the end of the file
        if current_start < len(content):
            final_text = content[current_start:len(content)]
            blocks.append((final_text, current_start, len(content)))

        acc_text: str = ""
        current_start = 0
        current_end: int = 0

    def process_file(self, file_path: Path) -> list[MinimalSource]:
        """Reads a file and delegates it to the correct parser.

        Args:
            file_path (Path): The path to the file to be processed.

        Returns:
            list[MinimalSource]: A list of validated chunk metadata sources.
        """
        try:
            if file_path.suffix == ".py":
                raw_chunks = self._process_py(file_path)
            elif file_path.suffix == ".md":
                raw_chunks = self._process_md(file_path)
            else:
                return []

            # Package the raw chunks into Pydantic models
            sources = []
            for text, start, end in raw_chunks:
                sources.append(
                    MinimalSource(
                        file_path=str(file_path),
                        first_character_index=start,
                        last_character_index=end
                    )
                )
            return sources

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
