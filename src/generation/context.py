"""Utilities for extracting and formatting textual context from sources."""

import functools
from pathlib import Path
from src.models import MinimalSource


@functools.lru_cache(maxsize=1024)
def _read_file_cached(file_path: str) -> str:
    """Reads and caches file content to prevent redundant disk I/O.

    Args:
        file_path (str): The absolute or relative path to the file.

    Returns:
        str: The raw string content of the file. Returns an empty
            string if the file cannot be found or read.
    """
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return ""


class ContextBuilder:
    """Builder for reconstructing and formatting context from sources."""

    def build(
        self, sources: list[MinimalSource], max_context_size: int = 3000
    ) -> str:
        """Constructs a formatted context string from retrieved sources.

        Extracts the exact character slices from the cached files and
        prepends each chunk with structural attribution metadata.

        Args:
            sources (list[MinimalSource]): List of retrieved chunk metadata.
            max_context_size (int): Maximum character limit for output
                context. Defaults to 3000.

        Returns:
            str: The aggregated and formatted context ready for an LLM.
        """
        if not sources:
            return ""

        formatted_chunks: list[str] = []

        for i, source in enumerate(sources, start=1):
            content = _read_file_cached(source.file_path)
            if not content:
                continue

            start = source.first_character_index
            end = source.last_character_index
            chunk_text = content[start:end]

            # Format the header trail if it exists
            trail = " > ".join(source.context_headers)
            header_str = f" | Section: {trail}" if trail else ""

            # Assemble the attribution block
            block = (
                f"--- [Source {i}: {source.file_path}{header_str}] ---\n"
                f"{chunk_text.strip()}\n"
            )
            formatted_chunks.append(block)

        full_context = "\n".join(formatted_chunks)

        if len(full_context) <= max_context_size:
            return full_context

        truncated = full_context[:max_context_size]
        cut_index = truncated.rfind("\n\n")

        if cut_index == -1:
            cut_index = max_context_size

        return truncated[:cut_index]
