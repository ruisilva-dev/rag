from pathlib import Path
from typing import Callable
import sys
import re
import ast
from rag.models import MinimalSource


ChunkData = list[tuple[str, int, int, list[str]]]


class FileChunker:
    """A text-splitting engine for Python code and Markdown files.

    Attributes:
        max_chunk_size (int): The maximum character length permitted
            for an individual text chunk.
        parsers (dict[str, Callable[[Path], ChunkData]]): A dispatch mapping of
            supported file extensions to their respective processing methods.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes the FileChunker with a maximum chunk size.

        Args:
            max_chunk_size (int): Maximum characters per chunk.
                Defaults to 2000.
        """
        self.max_chunk_size: int = max_chunk_size
        self.parsers: dict[str, Callable[[Path], ChunkData]] = {
            ".py": self._process_py,
            ".md": self._process_md
        }

    def _extract_blocks(
        self,
        content: str,
        pattern: str,
        lstrip_chars: str | None,
        clean_func: Callable[[str, str], str],
        clean_chars: str
    ) -> ChunkData:
        """Extracts text blocks based on structural boundaries.

        Scans the file text using a regex pattern to identify headers or
        signatures, tracks depth levels, and cleanly segments content
        while maintaining a hierarchical context stack.

        Args:
            content: The raw string content of the target file to slice.
            pattern: The regular expression used to find structural
                demarcations.
            lstrip_chars: Substring used to determine hierarchy depth.
                Pass None to strip standard whitespace (Python mode).
            clean_func: A string utility method to filter signatures.
            clean_chars: Characters targeted for cleaning.

        Returns:
            A list of structural text blocks, where each entry is a
            tuple containing the slice text, absolute start index,
            absolute end index, and active context headers.
        """
        current_start: int = 0
        active_headers: list[tuple[int, str]] = []

        blocks: ChunkData = []

        # Slice content between function/class definitions
        for match in re.finditer(pattern, content, flags=re.MULTILINE):
            # Handle past block
            block_text: str = content[current_start:match.start()]
            context_headers: list[str] = [
                header[1] for header in active_headers
            ]
            blocks.append(
                (block_text, current_start, match.start(), context_headers)
            )

            # Extract details for new match
            line_end: int = content.find("\n", match.start())
            if line_end == -1:
                line_end = len(content)
            line: str = content[match.start():line_end]
            current_indent: int = len(line) - len(line.lstrip(lstrip_chars))
            clean_line: str = clean_func(line, clean_chars)

            # Clear deeper or equal headers to remove sibling contexts
            while active_headers and active_headers[-1][0] >= current_indent:
                active_headers.pop()
            active_headers.append((current_indent, clean_line))

            # Update tracker
            current_start = match.start()

        # Cleanup for the end of the file
        if current_start < len(content):
            final_text = content[current_start:]
            context_headers = [header[1] for header in active_headers]
            blocks.append(
                (final_text, current_start, len(content), context_headers)
            )

        return blocks

    def _process_py(self, file_path: Path) -> ChunkData:
        content: str = file_path.read_text(encoding="utf-8")

        try:
            tree: ast.Module = ast.parse(content)
        except (ValueError, SyntaxError) as e:
            print(
                f"Error when processing file {file_path}: {e}. ",
                "Returning an empty chunk for this file.",
                file=sys.stderr
            )
            return []

        line_starts: list[int] = [0]

        for i, line in enumerate(content.splitlines()):
            curr_start: int = line_starts[i]
            next_start: int = curr_start + len(line) + 1

            line_starts.append(next_start)

    def _process_md(self, file_path: Path) -> ChunkData:
        """Splits a Markdown file into logical text segments.

        Reads the file content and splits it by double newlines to isolate
        paragraphs. Greedily packs these paragraphs into chunks. If a single
        paragraph exceeds the maximum size, it falls back to line-by-line
        splitting. If any individual line still exceeds the maximum size, it
        applies a final character-by-character subdivision.

        Args:
            file_path (Path): The path to the Markdown file.

        Returns:
            ChunkData: A list of chunks, where each tuple contains the chunk
                text, the absolute starting character index, and the absolute
                ending character index.
        """
        content: str = file_path.read_text(encoding="utf-8")
        pattern: str = r"^#+ "

        blocks: ChunkData = self._extract_blocks(
            content, pattern, "#", str.lstrip, "# "
        )

        if not blocks:
            return []

        chunks: ChunkData = self._pack_blocks(blocks, "\n\n")

        return chunks

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
        try:
            parser = self.parsers.get(file_path.suffix)
            if parser is None:
                return []

            raw_chunks = parser(file_path)

            # Package the raw chunks into Pydantic models
            sources = []
            for _, start, end, headers in raw_chunks:
                sources.append(
                    MinimalSource(
                        file_path=str(file_path),
                        first_character_index=start,
                        last_character_index=end,
                        context_headers=headers
                    )
                )
            return sources

        except (OSError, UnicodeDecodeError) as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(
                f"Unexpected error processing file {file_path}: {e}",
                file=sys.stderr
            )
            return []
