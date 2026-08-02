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
    return BM25SearchEngine(bm25s.BM25(), bm25s.BM25(), [], [])


def test_tokenizer() -> None:
    """Verifies that the tokenizer handles code syntax and edge cases.

    Ensures that text normalization lowercases strings, handles whitespace
    variations, ignores pure punctuation, and processes snake_case and
    camelCase based on the domain flag.
    """
    assert tokenize("Hello; there. snake_case?", is_code=False) == [
        "hello", "there", "snake_case"
    ]
    assert tokenize("Hello; there. snake_case?", is_code=True) == [
        "hello", "there", "snake", "case"
    ]
    assert tokenize("python3 version_2", is_code=False) == [
        "python3", "version_2"
    ]
    assert tokenize(";;;    !!!", is_code=False) == []
    assert tokenize("hello\tworld\npython   code", is_code=False) == [
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
    searcher.indices["code"].retriever.index([["dummy"]])
    searcher.indices["docs"].retriever.index([["dummy"]])

    batch_result = searcher.batch_search(
        questions=[],
        limit=5
    )

    assert batch_result.k == 5
    assert batch_result.search_results == []


def test_search_accuracy(indexer: BM25Indexer, tmp_path: Path) -> None:
    """Validates that search ranks relevant documents over irrelevant ones.

    Ensures that the trained BM25 retrieval engine successfully selects the
    correct source metadata when given a targeted query string.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    mock_file_acc = tmp_path / "accurate.md"
    mock_file_acc_content = (
        "This function establishes a secure database "
        "connection using a postgreSQL client."
    )
    mock_file_inacc = tmp_path / "inaccurate.md"
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

    docs_retriever = bm25s.BM25()
    docs_retriever.index(corpus)
    code_retriever = bm25s.BM25()
    code_retriever.index([["dummy"]])

    searcher = BM25SearchEngine(
        code_retriever=code_retriever,
        docs_retriever=docs_retriever,
        code_sources=[],
        docs_sources=indexer.indexed_sources
    )

    most_accurate = searcher.search("database connection", limit=1)

    assert most_accurate[0] == accurate_source
    assert len(most_accurate) == 1


def test_search_excessive_limit(
    indexer: BM25Indexer, tmp_path: Path
) -> None:
    """Verifies that searching with a limit higher than corpus size behaves.

    Ensures that when the requested limit exceeds the total number of indexed
    documents, the engine gracefully handles padding values instead of
    duplicating the final document.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
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

    docs_retriever = bm25s.BM25()
    docs_retriever.index(corpus)
    code_retriever = bm25s.BM25()
    code_retriever.index([["dummy"]])

    searcher = BM25SearchEngine(
        code_retriever=code_retriever,
        docs_retriever=docs_retriever,
        code_sources=[],
        docs_sources=indexer.indexed_sources
    )

    results = searcher.search("data science", limit=5)

    assert len(results) == 1
    assert len(results) == len(indexer.indexed_sources)
    assert results[0] == source


def test_search_to_model(indexer: BM25Indexer, tmp_path: Path) -> None:
    """Validates that search_to_model returns the correct Pydantic object.

    Ensures that the resulting MinimalSearchResults object maps fields like
    question_id, query string, and retrieved sources properly.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
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

    code_retriever = bm25s.BM25()
    code_retriever.index(corpus)
    docs_retriever = bm25s.BM25()
    docs_retriever.index([["dummy"]])

    searcher = BM25SearchEngine(
        code_retriever=code_retriever,
        docs_retriever=docs_retriever,
        code_sources=indexer.indexed_sources,
        docs_sources=[]
    )

    result = searcher.search_to_model(
        question_id="q_single",
        query_string="calculate total",
        limit=1
    )

    assert result.question_id == "q_single"
    assert result.question == "calculate total"
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources[0] == source


def test_batch_search(indexer: BM25Indexer, tmp_path: Path) -> None:
    """Validates that batch_search aggregates queries into a container.

    Ensures that executing batch searches accurately returns a structured
    StudentSearchResults wrapper holding all individual query items.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
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

    docs_retriever = bm25s.BM25()
    docs_retriever.index(corpus)
    code_retriever = bm25s.BM25()
    code_retriever.index([["dummy"]])

    searcher = BM25SearchEngine(
        code_retriever=code_retriever,
        docs_retriever=docs_retriever,
        code_sources=[],
        docs_sources=indexer.indexed_sources
    )

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
    """Validates that sub-indices metadata survives disk serialization.

    Ensures that calling save correctly writes out the domain tracking arrays
    and that load_from_disk successfully reconstructs both sub-indices.

    Args:
        indexer: A pytest fixture providing a BM25Indexer instance.
        tmp_path: A pytest fixture providing a temporary directory path.
    """
    save_dir = tmp_path / "bm25_index"
    code_dir = save_dir / "code"
    docs_dir = save_dir / "docs"
    code_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    mock_code_source = MinimalSource(
        file_path="main.py",
        first_character_index=12,
        last_character_index=45
    )
    mock_docs_source = MinimalSource(
        file_path="readme.md",
        first_character_index=0,
        last_character_index=30
    )

    # Build & save code sub-index
    indexer.indexed_sources = [mock_code_source]
    code_retriever = bm25s.BM25()
    code_retriever.index([["def", "main"]])
    indexer.save(str(code_dir), code_retriever)

    # Build & save docs sub-index
    indexer.indexed_sources = [mock_docs_source]
    docs_retriever = bm25s.BM25()
    docs_retriever.index([["readme", "documentation"]])
    indexer.save(str(docs_dir), docs_retriever)

    # Reconstruct both sub-indices from disk
    searcher = BM25SearchEngine.load_from_disk(str(save_dir))

    assert len(searcher.indices["code"].sources) == 1
    assert searcher.indices["code"].sources[0] == mock_code_source
    assert len(searcher.indices["docs"].sources) == 1
    assert searcher.indices["docs"].sources[0] == mock_docs_source
