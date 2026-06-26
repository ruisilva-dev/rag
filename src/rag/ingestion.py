from pathlib import Path
from typing import Generator, Callable
import itertools
import re
from rag.models import MinimalSource


ChunkData = list[tuple[str, int, int]]


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
        self.repo_path: Path = Path(repo_path)

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
        self.max_chunk_size: int = max_chunk_size
        self.parsers: dict[str, Callable[[Path], ChunkData]] = {
            ".py": self._process_py,
            ".md": self._process_md
        }

    def _process_py(self, file_path: Path) -> ChunkData:
        """Splits a Python file into logical code blocks.

        Reads the file content and isolates functions and classes using regex.
        Greedily packs these blocks into chunks. If a single function exceeds
        the maximum size, it falls back to line-by-line splitting.

        Args:
            file_path (Path): The path to the Python file.

        Returns:
            ChunkData: A list of chunks with text,
                start index, and end index.
        """
        content: str = file_path.read_text(encoding="utf-8")
        pattern: str = r"^(def |class )"
        current_start: int = 0

        blocks: ChunkData = []

        # Slice content between function/class definitions
        for match in re.finditer(pattern, content, flags=re.MULTILINE):
            block_text: str = content[current_start:match.start()]
            blocks.append((block_text, current_start, match.start()))
            current_start = match.start()

        # Cleanup for the end of the file
        if current_start < len(content):
            final_text = content[current_start:]
            blocks.append((final_text, current_start, len(content)))

        if not blocks:
            return []

        # Greedy packing
        chunks: ChunkData = []
        acc_text: str = ""
        current_start = blocks[0][1]
        current_end: int = blocks[0][2]

        for block_text, block_start, block_end in blocks:
            # Handle oversized block line-by-line
            if len(block_text) > self.max_chunk_size:
                if acc_text:
                    chunks.append((acc_text, current_start, current_end))
                    acc_text = ""

                lines = block_text.split("\n")
                sub_acc = lines[0]
                sub_start = block_start
                sub_end = block_start + len(lines[0])
                local_offset = len(lines[0]) + 1

                for line in lines[1:]:
                    if len(sub_acc) + 1 + len(line) <= self.max_chunk_size:
                        sub_acc += "\n" + line
                        sub_end = sub_start + local_offset + len(line)
                    else:
                        chunks.append((sub_acc, sub_start, sub_end))
                        sub_acc = line
                        sub_start = block_start + local_offset
                        sub_end = sub_start + len(line)

                    local_offset += len(line) + 1

                chunks.append((sub_acc, sub_start, sub_end))
                continue

            # Check if block fits (+1 for '\n')
            if not acc_text:
                acc_text = block_text
                current_start = block_start
                current_end = block_end
            elif len(acc_text) + 1 + len(block_text) <= self.max_chunk_size:
                acc_text += "\n" + block_text
                current_end = block_end
            else:
                chunks.append((acc_text, current_start, current_end))
                acc_text = block_text
                current_start = block_start
                current_end = block_end

        if acc_text:
            chunks.append((acc_text, current_start, current_end))

        return chunks

    def _process_md(self, file_path: Path) -> ChunkData:
        """Splits a Markdown file into logical text segments.

        Reads the file content and splits it by double newlines to isolate
        paragraphs. It then greedily packs these paragraphs into chunks that
        respect the maximum chunk size, preserving exact character indices.

        Args:
            file_path (Path): The path to the Markdown file.

        Returns:
            ChunkData: A list of chunks, where each tuple
                contains the chunk text, the absolute starting character index,
                and the absolute ending character index.
        """
        content: str = file_path.read_text(encoding="utf-8")
        current_start: int = 0

        blocks: ChunkData = []

        # Slice content between matches (paragraphs)
        for match in re.finditer(r"\n\n", content):
            block_text: str = content[current_start:match.start()]
            blocks.append((block_text, current_start, match.start()))

            current_start = match.end()

        # Cleanup for the end of the file
        if current_start < len(content):
            final_text = content[current_start:len(content)]
            blocks.append((final_text, current_start, len(content)))

        if not blocks:
            return []

        # Greedy packing
        chunks: ChunkData = []

        acc_text: str = blocks[0][0]
        current_start = blocks[0][1]
        current_end: int = blocks[0][2]

        for block_text, block_start, block_end in blocks[1:]:
            # Check if block_text will exceed chunk capacity
            # Add 2 for "\n\n" that was skipped earlier and will be added now
            if len(acc_text) + 2 + len(block_text) <= self.max_chunk_size:
                acc_text += "\n\n" + block_text
                current_end = block_end
            else:
                # Chunk is full
                chunks.append((acc_text, current_start, current_end))

                acc_text = block_text
                current_start = block_start
                current_end = block_end

        # Save last accumulated chunk after loop end
        if acc_text:
            chunks.append((acc_text, current_start, current_end))
        return chunks

    def process_file(self, file_path: Path) -> list[MinimalSource]:
        """Reads a file and delegates it to the correct parser.

        Args:
            file_path (Path): The path to the file to be processed.

        Returns:
            list[MinimalSource]: A list of validated chunk metadata sources.
        """
        try:
            parser = self.parsers.get(file_path.suffix)
            if parser is None:
                return []

            raw_chunks = parser(file_path)

            # Package the raw chunks into Pydantic models
            sources = []
            for _, start, end in raw_chunks:
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
