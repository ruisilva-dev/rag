from abc import ABC, abstractmethod
from pathlib import Path
import sys

Chunk = tuple[str, int, int, list[str]]


class BaseParser(ABC):
    """Abstract base class for document and code parsers.

    Attributes:
        max_chunk_size: Maximum allowed character length for any chunk.
    """

    def __init__(self, max_chunk_size: int) -> None:
        """Initializes BaseParser with chunking size limit.

        Args:
            max_chunk_size: Maximum allowed character length per chunk.
        """
        self.max_chunk_size: int = max_chunk_size

    def _read_file(self, file_path: Path | str) -> str | None:
        """Reads content from a target file as UTF-8 text.

        Args:
            file_path: Path to the target file to be read.

        Returns:
            The string content of the file, or None if reading fails.
        """
        try:
            return Path(file_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(
                f"Warning: Skipping file '{file_path}': {e}", file=sys.stderr
            )
            return None
        except Exception as e:
            print(
                f"Warning: Unexpected error in '{file_path}': {e}",
                file=sys.stderr
            )

    def _handle_fallback_split(
        self,
        chunks: list[Chunk],
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

    def _pack_blocks(self, blocks: list[Chunk], sep: str) -> list[Chunk]:
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
        chunks: list[Chunk] = []
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

                # Add new headers only
                current_headers.extend([
                    header for header in headers
                    if header not in current_headers
                ])

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

    @abstractmethod
    def parse(self, file_path: Path) -> list[Chunk]:
        """Parses a given file into a list of contextual chunks.

        Args:
            file_path: Path object pointing to the file to parse.

        Returns:
            A list of parsed Chunk tuples with context metadata.
        """
        ...
