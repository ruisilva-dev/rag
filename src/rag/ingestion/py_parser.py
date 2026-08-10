from pathlib import Path
from rag.ingestion.base import BaseParser, Chunk
import ast
import sys


class PythonParser(BaseParser):
    """AST-based parser for Python source files."""

    def _traverse(
        self,
        node: ast.AST,
        active_headers: list[str],
        blocks: list[Chunk],
        content: str,
        line_starts: list[int]
    ) -> None:
        """Recursively traverses AST nodes to extract compliant blocks.

        Args:
            node: Current AST node being processed.
            active_headers: Hierarchical lineage of outer block headers.
            blocks: Accumulator list where extracted chunks are stored.
            content: Full string content of the source file.
            line_starts: List of starting character offsets per line.
        """
        # Check if node is a definition node
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            # Build context header based on specific node type
            if isinstance(node, ast.ClassDef):
                header = f"class {node.name}:"
            elif isinstance(node, ast.AsyncFunctionDef):
                header = f"async def {node.name}:"
            else:
                header = f"def {node.name}:"
            current_headers = active_headers + [header]

            # Verify optional end bounds are present before doing arithmetic
            if node.end_lineno is not None and node.end_col_offset is not None:
                # Map 1-based AST line/col to 0-based character index
                start_char = line_starts[node.lineno - 1] + node.col_offset
                end_char = (
                    line_starts[node.end_lineno - 1] + node.end_col_offset
                )

                # If node and child nodes fits the chunk, return immediately
                if (end_char - start_char) <= self.max_chunk_size:
                    text = content[start_char:end_char]
                    blocks.append(
                        (text, start_char, end_char, current_headers)
                    )
                    return
        else:
            current_headers = active_headers

        # Recursively traverse child nodes
        for child_node in ast.iter_child_nodes(node):
            self._traverse(
                child_node, current_headers, blocks, content, line_starts
            )

    def parse(self, file_path: Path) -> list[Chunk]:
        """Parses a Python source file into structured chunks via AST.

        Args:
            file_path: Path object pointing to the Python file.

        Returns:
            A list of parsed Chunk tuples with context headers.
        """
        content = self._read_file(file_path)
        if not content:
            return []

        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError, TypeError) as e:
            print(
                f"Warning: AST parse failed for '{file_path}': {e}",
                file=sys.stderr
            )
            return []
        except Exception as e:
            print(
                f"Warning: Unexpected error in '{file_path}': {e}",
                file=sys.stderr
            )
            return []

        line_starts: list[int] = [0]
        for i, line in enumerate(content.splitlines()):
            curr_start: int = line_starts[i]
            next_start: int = curr_start + len(line) + 1

            line_starts.append(next_start)

        blocks: list[Chunk] = []
        self._traverse(tree, [], blocks, content, line_starts)

        return self._pack_blocks(blocks, "\n")
