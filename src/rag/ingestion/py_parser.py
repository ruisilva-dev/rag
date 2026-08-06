import ast
from pathlib import Path
from rag.ingestion.base import BaseParser, Chunk


class PythonParser(BaseParser):
    def _traverse(self, node: ast.AST, active_headers: list[str]) -> None:
        new_header: str | None = None

        # Define new header signature for definition nodes
        if isinstance(node, ast.ClassDef):
            new_header = f"class {node.name}:"
        elif isinstance(node, ast.AsyncFunctionDef):
            new_header = f"async def {node.name}:"
        elif isinstance(node, ast.FunctionDef):
            new_header = f"def {node.name}:"

        if new_header:
            current_headers = active_headers + [new_header]
        else:
            current_headers = active_headers

        for child_node in ast.iter_child_nodes(node):
            self._traverse(child_node, current_headers)

    def parse(self, file_path: Path) -> list[Chunk]:
        try:
            content = file_path.read_text(encoding="utf-8")