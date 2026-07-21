from pathlib import Path
from rag.ingestion import FileDiscoverer, FileChunker


def test_file_discovery(tmp_path: Path) -> None:
    """Verifies that only supported extensions are discovered.

    Ensures that the file discoverer identifies Python (.py) and Markdown
    (.md) files while completely ignoring unsupported files like text
    (.txt) files.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_py = tmp_path / "test.py"
    mock_md = tmp_path / "test.md"
    mock_txt = tmp_path / "test.txt"

    mock_py.write_text("print('hello')", encoding="utf-8")
    mock_md.write_text("# Hello", encoding="utf-8")
    mock_txt.write_text("Hello", encoding="utf-8")

    discoverer = FileDiscoverer(tmp_path)

    file_number = len(list(discoverer.discover_files()))

    assert file_number == 2


def test_single_chunk_py(tmp_path: Path) -> None:
    """Validates greedy packing of small Python functions into a single chunk.

    Verifies that multiple function blocks are grouped together when their
    combined text length fits entirely within the max_chunk_size limit.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "def func_one():\n"
        "    return 1\n\n"
        "def func_two():\n"
        "    return 2\n"
    )

    mock_py = tmp_path / "small_code.py"
    mock_py.write_text(content, encoding="utf-8")

    chunker = FileChunker(max_chunk_size=2000)
    sources = chunker.process_file(mock_py)

    assert len(sources) == 1
    assert sources[0].first_character_index == 0
    assert sources[0].last_character_index == len(content)


def test_single_chunk_md(tmp_path: Path) -> None:
    """Validates greedy packing of short Markdown paragraphs into one chunk.

    Ensures that separate paragraphs separated by double newlines are
    aggregated into one chunk if they do not exceed the max_chunk_size limit.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "# Header\n\n"
        "This is the first paragraph.\n\n"
        "This is the second paragraph."
    )

    mock_md = tmp_path / "some_text.md"
    mock_md.write_text(content, encoding="utf-8")

    chunker = FileChunker(max_chunk_size=2000)
    sources = chunker.process_file(mock_md)

    assert len(sources) == 1
    assert sources[0].first_character_index == 0
    assert sources[0].last_character_index == len(content)


def test_multi_chunk_py(tmp_path: Path) -> None:
    """Verifies Python block splitting when cumulative size exceeds limits.

    Ensures that when a new function block cannot fit into the current
    chunk, the chunker seals the current chunk and starts a new one.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    func_one = "def small_func_one():\n    pass\n\n"
    func_two = "def small_func_two():\n    pass\n"
    content = func_one + func_two

    mock_py = tmp_path / "multi_chunk.py"
    mock_py.write_text(content, encoding="utf-8")

    # Force split between functions
    chunker = FileChunker(max_chunk_size=len(func_one))
    sources = chunker.process_file(mock_py)

    assert len(sources) == 2
    assert sources[0].first_character_index == 0
    assert sources[0].last_character_index == len(func_one)
    assert sources[1].first_character_index == len(func_one)
    assert sources[1].last_character_index == len(content)


def test_multi_chunk_md(tmp_path: Path) -> None:
    """Verifies Markdown paragraph splitting and context header tracking.

    Validates that separate paragraphs are allocated into separate chunks if
    combining them violates size constraints, and ensures each chunk retains
    its specific active section headers.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "# Section One\n"
        "Content for the first section.\n"
        "# Section Two\n"
        "Content for the second section."
    )

    mock_md = tmp_path / "multi_chunk.md"
    mock_md.write_text(content, encoding="utf-8")

    # Force split between headers
    chunker = FileChunker(max_chunk_size=60)
    sources = chunker.process_file(mock_md)

    assert len(sources) == 2
    assert sources[0].first_character_index == 0
    assert sources[0].context_headers == ["Section One"]
    assert sources[1].context_headers == ["Section Two"]


def test_chunk_fallback_line_py(tmp_path: Path) -> None:
    """Tests line-by-line fallback and context preservation for Python.

    Verifies that an oversized code block is partitioned into line-based
    sub-chunks under the size limit, and ensures that every sub-chunk
    correctly retains the parent class or function context headers.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "def giant_function_definition():\n"
        "    line_one_of_body = 1\n"
        "    line_two_of_body = 2\n"
        "    line_three_of_body = 3\n"
    )
    mock_py = tmp_path / "fallback.py"
    mock_py.write_text(content, encoding="utf-8")

    # Set tiny max chunk size to force line-by-line splitting
    chunker = FileChunker(max_chunk_size=40)
    sources = chunker.process_file(mock_py)

    max_size = chunker.max_chunk_size

    assert len(sources) > 1
    assert sources[0].first_character_index == 0
    assert sources[-1].last_character_index == len(content)
    assert all(
        (s.last_character_index - s.first_character_index) <= max_size
        for s in sources
    )
    assert all(
        s.context_headers == ["def giant_function_definition()"]
        for s in sources
    )


def test_chunk_fallback_char_py(tmp_path: Path) -> None:
    """Tests the character-by-character strategy for giant Python lines.

    Verifies that a continuous string of code or text lacking any newline
    characters is successfully partitioned into sub-chunks matching the
    exact max_chunk_size constraint.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = ("ThisIsAContinuousStringThatIsLongerThanFortyCharacters")
    mock_py = tmp_path / "fallback.py"
    mock_py.write_text(content, encoding="utf-8")

    # Set tiny max chunk size to force char-by-char splitting
    chunker = FileChunker(max_chunk_size=40)
    sources = chunker.process_file(mock_py)

    max_size = chunker.max_chunk_size

    assert len(sources) > 1
    assert sources[0].first_character_index == 0
    assert sources[-1].last_character_index == len(content)
    assert all(
        (s.last_character_index - s.first_character_index) <= max_size
        for s in sources
    )


def test_chunk_fallback_line_md(tmp_path: Path) -> None:
    """Tests line-by-line fallback and context preservation for Markdown.

    Verifies that an oversized paragraph block is partitioned into line-based
    sub-chunks under the size limit, and ensures that every sub-chunk
    correctly retains the parent section context headers.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "# Documentation\n"
        "Short line one.\n"
        "Short line two.\n"
        "Short line three."
    )
    mock_md = tmp_path / "fallback.md"
    mock_md.write_text(content, encoding="utf-8")

    chunker = FileChunker(max_chunk_size=30)
    sources = chunker.process_file(mock_md)

    max_size = chunker.max_chunk_size

    assert len(sources) > 1
    assert sources[0].first_character_index == 0
    assert sources[-1].last_character_index == len(content)
    assert all(
        (s.last_character_index - s.first_character_index) <= max_size
        for s in sources
    )
    assert all(s.context_headers == ["Documentation"] for s in sources)


def test_chunk_fallback_char_md(tmp_path: Path) -> None:
    """Tests the character-by-character strategy for giant Markdown lines.

    Verifies that an exceptionally long continuous text line with no
    newline characters is forcefully subdivided into chunks matching the
    exact max_chunk_size constraint.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "This is an exceptionally long single paragraph block string designed "
        "to run way past the maximum configured size parameter constraint "
        "boundary rule set."
    )
    mock_md = tmp_path / "fallback.md"
    mock_md.write_text(content, encoding="utf-8")

    chunker = FileChunker(max_chunk_size=30)
    sources = chunker.process_file(mock_md)

    max_size = chunker.max_chunk_size

    assert len(sources) > 1
    assert sources[0].first_character_index == 0
    assert sources[-1].last_character_index == len(content)
    assert all(
        (s.last_character_index - s.first_character_index) <= max_size
        for s in sources
    )


def test_chunk_error_py(tmp_path: Path) -> None:
    """Verifies error handling when a Python file does not exist.

    Ensures that file system errors are intercepted safely, logging the
    error and returning an empty source list instead of crashing.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    non_existent_file = tmp_path / "ghost_file.py"
    chunker = FileChunker()

    sources = chunker.process_file(non_existent_file)
    assert sources == []


def test_chunk_error_md(tmp_path: Path) -> None:
    """Verifies error handling when a Markdown file does not exist.

    Ensures that processing a missing documentation file returns an empty
    list cleanly and protects pipeline stability.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    non_existent_file = tmp_path / "ghost_file.md"
    chunker = FileChunker()

    sources = chunker.process_file(non_existent_file)
    assert sources == []


def test_context_headers_py(tmp_path: Path) -> None:
    """Verifies Python class and method hierarchies are tracked in metadata.

    Ensures that the indentation-based parsing successfully identifies nested
    classes and methods, pushing them onto the active headers stack so that
    chunks retain their complete structural lineage.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "class DataModel:\n"
        "    def entry_point(self):\n"
        "        x = 1\n"
    )
    mock_py = tmp_path / "context_test.py"
    mock_py.write_text(content, encoding="utf-8")

    chunker = FileChunker(max_chunk_size=2000)
    sources = chunker.process_file(mock_py)

    # The final chunk contains the body "x = 1"
    assert len(sources) == 1
    assert sources[0].context_headers == [
        "class DataModel", "    def entry_point(self)"
    ]


def test_context_headers_md(tmp_path: Path) -> None:
    """Verifies Markdown header nesting and popping hierarchies are tracked.

    Ensures that deeper header levels are appended to the context stack, and
    that returning to a higher-level header correctly pops out any deeper,
    obsolete headings.

    Args:
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    content = (
        "# Setup\n"
        "Content here.\n"
        "## Installation\n"
        "More content here.\n"
        "# Usage\n"
        "Run the main script."
    )
    mock_md = tmp_path / "context_test.md"
    mock_md.write_text(content, encoding="utf-8")

    # Use a small chunk size to force each section into its own chunk
    chunker = FileChunker(max_chunk_size=40)
    sources = chunker.process_file(mock_md)

    assert len(sources) >= 3
    # The first chunk under '# Setup'
    assert sources[0].context_headers == ["Setup"]
    # The chunk that reaches the end under '# Usage'
    assert sources[-1].context_headers == ["Usage"]
