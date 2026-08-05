import ast


class PythonParser:
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size: int = max_chunk_size

    def _traverse(self, node: ast.AST, active_headers: list[str]) -> None:
        ...