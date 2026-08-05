from pathlib import Path
from typing import Generator, Callable
import sys
import itertools
import re
import ast
from rag.models import MinimalSource


ChunkData = list[tuple[str, int, int, list[str]]]


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

    def _handle_fallback_split(
        self,
        chunks: ChunkData,
        block_text: str,
        block_start: int,
        headers: list[str]
    ) -> None:
        """Splits an oversized text block using fallback strategies.

        Partitions a block line-by-line into compliant lengths. If any
        individual line still breaks the limit, it initiates a terminal
        character-by-character division loop.

        Args:
            chunks: Accumulator collection where sub-chunks are added.
            block_text: The oversized string section to partition.
            block_start: The absolute starting position index.
            headers: Contextual lineage elements tied to this block.
        """
        lines: list[str] = block_text.split("\n")
        sub_acc: str = ""
        local_offset: int = 0
        sub_start: int = block_start
        sub_end: int = block_start

        for line in lines:
            line_len: int = len(line)

            # Handle oversized line, character-by-chracter
            if line_len > self.max_chunk_size:
                # Save anything accumulated up until now
                if sub_acc:
                    chunks.append(
                        (sub_acc, sub_start, sub_end, headers)
                    )
                    local_offset += len(sub_acc) + 1
                    sub_acc = ""

                # Slice the line character-by-chracter
                for i in range(0, line_len, self.max_chunk_size):
                    curr_slice: str = line[i:i + self.max_chunk_size]
                    slice_start: int = block_start + local_offset + i
                    slice_end: int = slice_start + len(curr_slice)

                    chunks.append(
                        (curr_slice, slice_start, slice_end, headers)
                    )

                local_offset += line_len + 1
                continue

            # Seed empty buffer
            if not sub_acc:
                sub_start = block_start + local_offset
                sub_acc = line
                sub_end = sub_start + line_len

            # Combine into buffer if within max_chunk_size
            elif len(sub_acc) + 1 + line_len <= self.max_chunk_size:
                sub_acc += "\n" + line
                sub_end = sub_start + len(sub_acc)

            # Chunk is full
            else:
                chunks.append(
                    (sub_acc, sub_start, sub_end, headers)
                )
                local_offset += len(sub_acc) + 1
                sub_start = block_start + local_offset
                sub_acc = line
                sub_end = sub_start + line_len

        if sub_acc:
            chunks.append(
                (sub_acc, sub_start, sub_end, headers)
            )

    def _pack_blocks(self, blocks: ChunkData, sep: str) -> ChunkData:
        """Consolidates blocks greedily up to the maximum chunk limit.

        Packs sequential blocks together using an explicit delimiter. If
        a block causes the accumulator to pass the size boundary, the
        existing chunk is sealed. Completely oversized items fall back
        to division.

        Args:
            blocks: Extracted raw logical segments ready for bundling.
            sep: The delimiter sequence used to concatenate elements.

        Returns:
            A list of formatted chunks adhering to the threshold rules.
        """
        chunks: ChunkData = []
        acc_text: str = ""
        current_start = 0
        current_end: int = 0
        current_headers: list[str] = []
        # Default headers value for final cleanup
        headers: list[str] = []

        for block_text, block_start, block_end, headers in blocks:
            # Handle oversized block line-by-line
            if len(block_text) > self.max_chunk_size:
                # Save anything accumulated up until now
                if acc_text:
                    chunks.append(
                        (acc_text, current_start, current_end, current_headers)
                    )
                    acc_text = ""

                self._handle_fallback_split(
                    chunks, block_text, block_start, headers
                )

                continue

            # Seed empty buffer
            if not acc_text:
                acc_text = block_text
                current_start = block_start
                current_end = block_end
                current_headers = headers

            # Combine into buffer if within max_chunk_size
            elif (
                len(acc_text) + len(sep) + len(block_text)
                <= self.max_chunk_size
            ):
                acc_text += sep + block_text
                current_end = block_end
                current_headers = headers

            # Chunk is full
            else:
                chunks.append(
                    (acc_text, current_start, current_end, current_headers)
                )
                acc_text = block_text
                current_start = block_start
                current_end = block_end
                current_headers = headers

        # Cleanup
        if acc_text:
            chunks.append(
                (acc_text, current_start, current_end, current_headers)
            )

        return chunks

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
