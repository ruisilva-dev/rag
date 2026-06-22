from pathlib import Path
from typing import Generator
import itertools


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
