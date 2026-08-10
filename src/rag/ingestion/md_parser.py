from pathlib import Path

from rag.ingestion.base import BaseParser, Chunk


class MarkdownParser(BaseParser):
    """Header-aware parser for Markdown documentation files."""

    def parse(self, file_path: Path) -> list[Chunk]:
        """Parses a Markdown file into structured chunks based on headers.

        Args:
            file_path: Path object pointing to the Markdown file.

        Returns:
            A list of parsed Chunk tuples with active header metadata.
        """
        content = self._read_file(file_path)
        if not content:
            return []

        # Tracking variables
        in_code_block: bool = False
        headers_stack: list[tuple[int, str]] = []
        current_start: int = 0
        blocks: list[Chunk] = []

        line_offset = 0
        for line in content.split("\n"):
            line_len = len(line) + 1  # includes the '\n'

            if line.startswith("```"):
                in_code_block = not in_code_block
            elif not in_code_block and line.lstrip().startswith("#"):
                clean_line = line.lstrip()
                level = len(clean_line) - len(clean_line.lstrip("#"))
                title = clean_line.strip("# ")
                prev_text = content[current_start:line_offset]
                prev_start = current_start
                current_start = line_offset

                while headers_stack and headers_stack[-1][0] >= level:
                    headers_stack.pop()

                active_headers = [header[1] for header in headers_stack]
                headers_stack.append((level, title))

                if prev_text:
                    blocks.append(
                        (prev_text, prev_start, current_start, active_headers)
                    )

            line_offset += line_len

        # Cleanup
        final_text = content[current_start:]
        if final_text:
            active_headers = [header[1] for header in headers_stack]
            blocks.append(
                (final_text, current_start, len(content), active_headers)
            )

        return self._pack_blocks(blocks, "\n\n")
