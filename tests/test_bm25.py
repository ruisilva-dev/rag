import pytest
import bm25s
from pathlib import Path
from rag.bm25.indexer import BM25Indexer
from rag.bm25.engine import BM25SearchEngine
from rag.bm25.utils import tokenize
from rag.models import MinimalSource, UnansweredQuestion


@pytest.fixture
def indexer() -> BM25Indexer:
    """Provides a fresh instance of BM25Indexer for each test case.

    Returns:
        BM25Indexer: An uninitialized indexer instance.
    """
    return BM25Indexer()


@pytest.fixture
def searcher() -> BM25SearchEngine:
    """Provides a fresh instance of BM25SearchEngine for each test case.

    Returns:
        BM25SearchEngine: An uninitialized searcher instance.
    """
    return BM25SearchEngine(bm25s.BM25(), [])


def test_tokenizer() -> None:
    """Verifies that the tokenizer handles code syntax and edge cases.

    Ensures that text normalization lowercases strings, handles whitespace
    variations, ignores pure punctuation, and preserves snake_case
    identifiers.
    """
    assert tokenize("Hello; there. snake_case?") == [
        "hello", "there", "snake_case"
    ]
    assert tokenize("python3 version_2") == ["python3", "version_2"]
    assert tokenize(";;;    !!!") == []
    assert tokenize("hello\tworld\npython   code") == [
        "hello", "world", "python", "code"
    ]


def test_build_corpus(indexer: BM25Indexer, tmp_path: Path) -> None:
    """Verifies that building the corpus clears stale state and aligns sources.

    Ensures that any pre-existing source metadata is wiped from the indexer
    and that the newly generated tokenized corpus perfectly mirrors the
    fresh source input boundaries.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_stale_file = tmp_path / "path"
    mock_stale_file.write_text("stale", encoding="utf-8")
    mock_fresh_file = tmp_path / "new_path"
    mock_fresh_file.write_text("fresh", encoding="utf-8")

    stale_source = MinimalSource(
        file_path=str(mock_stale_file),
        first_character_index=0,
        last_character_index=1
    )
    fresh_source = MinimalSource(
        file_path=str(mock_fresh_file),
        first_character_index=0,
        last_character_index=5
    )

    indexer.indexed_sources.append(stale_source)

    corpus = indexer.build_corpus([fresh_source])

    assert indexer.indexed_sources[0] != stale_source
    assert indexer.indexed_sources[0] == fresh_source
    assert len(indexer.indexed_sources) == 1
    assert corpus == [["fresh"]]


def test_build_corpus_multi_chunk(
    indexer: BM25Indexer, tmp_path: Path
) -> None:
    """Verifies build_corpus correctly orders multiple chunks from one file.

    Ensures that when a single file contains multiple text boundaries, the
    extracted documents and tracking metadata maintain their relative order
    perfectly.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_file = tmp_path / "multi_chunk.txt"
    mock_file.write_text("hello world", encoding="utf-8")

    source_one = MinimalSource(
        file_path=str(mock_file),
        first_character_index=0,
        last_character_index=5
    )
    source_two = MinimalSource(
        file_path=str(mock_file),
        first_character_index=6,
        last_character_index=11
    )

    corpus = indexer.build_corpus([source_one, source_two])

    assert len(indexer.indexed_sources) == 2
    assert len(indexer.indexed_sources) == len(corpus)
    assert corpus == [["hello"], ["world"]]
    assert indexer.indexed_sources[0] == source_one
    assert indexer.indexed_sources[1] == source_two


def test_build_corpus_empty(indexer: BM25Indexer) -> None:
    """Verifies that build_corpus handles an empty source list gracefully.

    Ensures that passing an empty list resets the internal tracking state
    and returns an empty corpus list without crashing.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
    """
    indexer.indexed_sources = [
        MinimalSource(
            file_path="stale.py",
            first_character_index=0,
            last_character_index=10
        )
    ]

    corpus = indexer.build_corpus([])

    assert corpus == []
    assert indexer.indexed_sources == []


def test_batch_search_empty(searcher: BM25SearchEngine) -> None:
    """Verifies that batch_search handles an empty question list gracefully.

    Ensures that passing an empty list of questions returns a valid, empty
    StudentSearchResults container.

    Args:
        searcher: A pytest fixture providing a BM25SearchEngine instance.
    """
    searcher.retriever.index([["dummy"]])

    batch_result = searcher.batch_search(
        questions=[],
        limit=5
    )

    assert batch_result.k == 5
    assert batch_result.search_results == []


def test_search_accuracy(
    indexer: BM25Indexer, searcher: BM25SearchEngine, tmp_path: Path
) -> None:
    """Validates that search ranks relevant documents over irrelevant ones.

    Ensures that the trained BM25 retrieval engine successfully selects the
    correct source metadata when given a targeted query string.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        searcher: A pytest fixture providing a BM25SearchEngine instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_file_acc = tmp_path / "accurate"
    mock_file_acc_content = (
        "This function establishes a secure database "
        "connection using a postgreSQL client."
    )
    mock_file_inacc = tmp_path / "inaccurate"
    mock_file_inacc_content = (
        "A utility function to render the user profile avatar graphics on the "
        "dashboard."
    )

    mock_file_acc.write_text(mock_file_acc_content, encoding="utf-8")
    mock_file_inacc.write_text(mock_file_inacc_content, encoding="utf-8")

    accurate_source = MinimalSource(
        file_path=str(mock_file_acc),
        first_character_index=0,
        last_character_index=len(mock_file_acc_content)
    )
    inaccurate_source = MinimalSource(
        file_path=str(mock_file_inacc),
        first_character_index=0,
        last_character_index=len(mock_file_inacc_content)
    )

    corpus = indexer.build_corpus([accurate_source, inaccurate_source])

    searcher.indexed_sources = indexer.indexed_sources
    searcher.retriever.index(corpus)

    most_accurate = searcher.search("database connection", 1)

    assert most_accurate[0] == accurate_source
    assert len(most_accurate) == 1


def test_search_excessive_limit(
    indexer: BM25Indexer, searcher: BM25SearchEngine, tmp_path: Path
) -> None:
    """Verifies that searching with a limit higher than corpus size behaves.

    Ensures that when the requested limit exceeds the total number of indexed
    documents, the engine gracefully ignores internal padding values (-1)
    instead of duplicating the final document.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        searcher: A pytest fixture providing a BM25SearchEngine instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_file = tmp_path / "single.txt"
    mock_file.write_text("unique data science content", encoding="utf-8")

    source = MinimalSource(
        file_path=str(mock_file),
        first_character_index=0,
        last_character_index=27
    )

    corpus = indexer.build_corpus([source])

    searcher.indexed_sources = indexer.indexed_sources
    searcher.retriever.index(corpus)

    # Request a limit of 5 when only 1 document exists
    results = searcher.search("data science", limit=5)

    assert len(results) == 1
    assert len(results) == len(indexer.indexed_sources)
    assert results[0] == source


def test_search_to_model(
    indexer: BM25Indexer, searcher: BM25SearchEngine, tmp_path: Path
) -> None:
    """Validates that search_to_model returns the correct Pydantic object.

    Ensures that the resulting MinimalSearchResults object maps fields like
    question_id, query string, and retrieved sources properly.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        searcher: A pytest fixture providing a BM25SearchEngine instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_file = tmp_path / "model_test.py"
    content = "def calculate_total(): return 40"
    mock_file.write_text(content, encoding="utf-8")

    source = MinimalSource(
        file_path=str(mock_file),
        first_character_index=0,
        last_character_index=len(content)
    )

    corpus = indexer.build_corpus([source])

    searcher.indexed_sources = indexer.indexed_sources
    searcher.retriever.index(corpus)

    result = searcher.search_to_model(
        question_id="q_single",
        query_string="calculate total",
        limit=1
    )

    assert result.question_id == "q_single"
    assert result.question == "calculate total"
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources[0] == source


def test_batch_search(
    indexer: BM25Indexer, searcher: BM25SearchEngine, tmp_path: Path
) -> None:
    """Validates that batch_search aggregates queries into a container.

    Ensures that executing batch searches accurately returns a structured
    StudentSearchResults wrapper holding all individual query items.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        searcher: A pytest fixture providing a BM25SearchEngine instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_file = tmp_path / "batch_test.md"
    content = "The quick brown fox jumps over the lazy dog."
    mock_file.write_text(content, encoding="utf-8")

    source = MinimalSource(
        file_path=str(mock_file),
        first_character_index=0,
        last_character_index=len(content)
    )

    corpus = indexer.build_corpus([source])

    searcher.indexed_sources = indexer.indexed_sources
    searcher.retriever.index(corpus)

    question = UnansweredQuestion(
        question_id="q_batch_1",
        question="quick brown fox"
    )

    batch_result = searcher.batch_search(
        questions=[question],
        limit=1
    )

    assert batch_result.k == 1
    assert len(batch_result.search_results) == 1

    single_result = batch_result.search_results[0]
    assert single_result.question_id == "q_batch_1"
    assert single_result.retrieved_sources[0] == source


def test_save_and_load(indexer: BM25Indexer, tmp_path: Path) -> None:
    """Validates that indexed_sources metadata survives disk serialization.

    Ensures that calling save correctly writes out the tracking array and
    that load_metadata successfully reconstructs the exact original state.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    save_dir = tmp_path / "index_storage"
    save_dir.mkdir()

    mock_source = MinimalSource(
        file_path="main.py",
        first_character_index=12,
        last_character_index=45
    )
    indexer.indexed_sources = [mock_source]

    # Initialize a dummy retriever to satisfy internal save constraints
    retriever = bm25s.BM25()
    retriever.index([["dummy", "tokens"]])

    # Serialize the state to the temporary directory
    indexer.save(str(save_dir), retriever)

    # Reconstruct the tracking state into a completely fresh indexer
    searcher = BM25SearchEngine.load_from_disk(str(save_dir))

    assert len(searcher.indexed_sources) == 1
    assert searcher.indexed_sources[0] == mock_source
