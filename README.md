*This project has been created as part of the 42 curriculum by ruisilva.*

# RAG against the machine

## Description

This project is a **Retrieval-Augmented Generation (RAG)** system that answers
natural-language questions about a source code repository.
It ingests the repository, builds a searchable BM25 index over intelligently
chunked Python and Markdown files, retrieves the most relevant snippets for a
given question, and generates a source-grounded answer using a local LLM
(`Qwen/Qwen3-0.6B`). A dedicated evaluation module measures retrieval quality
with a recall@k metric based on character-range overlap (IoU) against a
ground-truth dataset.

The whole pipeline — indexing, searching, answering, and evaluating — is
exposed through a single command-line interface built with `fire`.

## Instructions

**Requirements**: Python 3.10+, [`uv`](https://docs.astral.sh/uv/) as the
package/project manager.

```bash
make install   # uv sync — installs project dependencies
make run       # uv run python3 -m src — launches the CLI
make debug     # runs the CLI under pdb
make lint      # flake8 + mypy (standard flags)
make lint-strict  # flake8 + mypy --strict
make clean     # removes __pycache__, .mypy_cache, .pytest_cache
```

### CLI commands

All commands are available via `uv run python -m src <command> [args]`:

| Command | Purpose |
|---|---|
| `index` | Discover, chunk, and index the repository into a BM25 index |
| `search` | Search the index for a single query string |
| `search_dataset` | Run batch search over a JSON dataset of questions |
| `answer` | Retrieve context and generate an answer for a single query |
| `answer_dataset` | Generate answers for a batch of previously retrieved search results |
| `evaluate` | Compute recall@k of retrieval results against ground truth |

## System architecture

The pipeline is composed of independent, single-responsibility modules wired
together by the CLI (`src/__main__.py`):

1. **Ingestion** (`src/ingestion.py`) — `FileDiscoverer` walks the repository
   for `.py` and `.md` files; `FileChunker` splits each file into chunks.
2. **Indexing** (`src/bm25/indexer.py`) — `BM25Indexer` groups chunks by file,
   extracts their raw text, and builds a tokenized BM25 corpus; the resulting
   index and chunk metadata (`metadata.json`) are persisted to disk.
3. **Retrieval** (`src/bm25/engine.py`) — `BM25SearchEngine` loads the saved
   index and metadata, tokenizes queries (stemmed, English stopwords removed),
   and returns the top-k matching `MinimalSource` chunks, for a single query
   or a batch of questions.
4. **Context building** (`src/generation/context.py`) — `ContextBuilder`
   re-reads the exact character slices for retrieved sources (with file-level
   caching) and formats them into an attributed context block, truncated to a
   maximum size.
5. **Generation** (`src/generation/generator.py`) — `AnswerGenerator` loads
   `Qwen/Qwen3-0.6B` and produces a source-grounded answer from the built
   context, using a system prompt that enforces citation and forbids
   hallucination.
6. **Evaluation** (`src/evaluation.py`) — `SearchEvaluator` compares retrieved
   sources against ground-truth sources and reports recall@1/3/5/10.
7. **CLI** (`src/__main__.py`) — a `fire`-driven `CLI` class exposes `index`,
   `search`, `search_dataset`, `answer`, `answer_dataset`, and `evaluate`,
   each wrapped in a shared error handler (`src/utils.py`) that turns
   `OSError`, malformed JSON, pydantic validation errors, and other
   exceptions into clean, non-crashing CLI error messages.

All data interchange uses the pydantic models in `src/models.py`
(`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`,
`MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`,
`StudentSearchResultsAndAnswer`).

## Chunking strategy

Two dedicated strategies are implemented, both capped at a configurable
`max_chunk_size` (default 2000 characters, set via `--max_chunk_size` on
`index`):

- **Python (`.py`)** — files are split at `def`/`class` boundaries using a
  regex scan. An indentation-based stack tracks which enclosing
  class/function each block belongs to, so every chunk carries a
  `context_headers` trail (e.g. `["class Foo", "def bar"]`) for better
  retrieval signal and citation. Blocks are then greedily packed together up
  to `max_chunk_size`.
- **Markdown (`.md`)** — files are split at heading boundaries (`#`, `##`,
  ...), with the same heading-stack and greedy-packing logic, joined with a
  blank line instead of a newline.
- **Fallback splitting** — if a single logical block (a huge function, or a
  section with no sub-headings) still exceeds `max_chunk_size`, it is split
  line by line, and if a single line is still too long, it is split
  character by character. This guarantees no chunk ever exceeds the limit.

## Retrieval method

Retrieval uses **BM25** via the `bm25s` library:

- Corpus texts are built per chunk as `"{file_path}: {header_trail}\n{chunk_text}"`,
  so the file path and structural context contribute to the ranking, not just
  the raw code/text.
- Tokenization uses an English Snowball stemmer (`PyStemmer`) and the
  extended `bm25s` English stopword list, applied identically to the corpus
  and to queries.
- `search()` retrieves candidates and returns the top-k `MinimalSource`
  results, capped to the number of indexed chunks to avoid out-of-range
  requests on small indexes.
- `search_to_model` / `batch_search` wrap single and batch queries into the
  `MinimalSearchResults` / `StudentSearchResults` pydantic models.

## Performance analysis

The retrieval system achieves high recall scores across the given document and code datasets, comfortably exceeding the project's baseline minimum requirements (≥80% for docs, ≥50% for code):

| Metric | Docs questions | Code questions |
| :--- | :--- | :--- |
| Recall@1 | 63.0% | 44.0% |
| Recall@3 | 81.0% | 62.0% |
| Recall@5 | 86.0% | 70.0% |
| Recall@10 | 90.0% | 78.0% |

- Indexing time: ~5 seconds for ~1,941 source files (budget: ≤ 5 min)
- Cold start latency: ~28 seconds (budget: ≤ 60 s)
- Warm retrieval throughput (200 questions): ~6 seconds (budget: ≤ 90 s)

## Design decisions

- **Character-based context budget** — `ContextBuilder` truncates on a
  character count (default 3000) as a practical proxy for the LLM's token
  budget, cutting cleanly at a paragraph boundary rather than mid-sentence.
- **Structure-aware chunking** — using an indentation/heading stack instead
  of fixed-size windows keeps functions, classes, and Markdown sections
  intact wherever possible, and gives each chunk a header trail useful both
  for ranking and for citation in generated answers.
- **Deterministic generation** — `AnswerGenerator` uses `do_sample=False` so
  answers are reproducible given the same retrieved context.
- **Centralized CLI error handling** — a single `catch_cli_errors` decorator
  wraps every CLI command, translating I/O, JSON, and pydantic validation
  failures into clear stderr messages with a clean exit code instead of a
  raw traceback.

## Challenges faced

- Deciding how to keep code and Markdown chunks meaningfully attributed to
  their enclosing function/class or section without over-fragmenting large
  files — solved with the indentation/heading stack plus a greedy packer and
  a line/character fallback for oversized blocks.
- Fitting retrieved context into the small context window of
  `Qwen/Qwen3-0.6B` without truncating mid-source — solved by truncating on
  paragraph boundaries in `ContextBuilder`.
- Making the CLI resilient to malformed datasets and edge-case arguments
  (e.g. `k=0`, missing files) without crashing — solved with a shared
  exception-handling decorator around all CLI entry points.

## Example usage

```bash
# Index the repository
uv run python -m src index --repo_path data/raw/vllm-0.10.1 --max_chunk_size 2000

# Search for a single query
uv run python -m src search "How to configure OpenAI server?" --k 10

# Batch search a dataset of questions
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results

# Answer a single question with retrieved context
uv run python -m src answer "How to configure OpenAI server?" --k 10

# Generate answers for a whole batch of search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer

# Evaluate retrieval quality against ground truth
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_chunk_size 2000
```

## Resources

- [`bm25s` documentation](https://github.com/xhluca/bm25s)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire documentation](https://github.com/google/python-fire)
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)

### AI usage:

* **Assisted Tasks:** AI tools were utilized during the design phase to prototype the structure-aware chunking regex parsing logic, structure the Pydantic data models for compatibility with standard evaluation schemas, and explore optimal generation pipeline parameters with Hugging Face transformers.  
* **Human Responsibility:** All AI-generated code snippets and templates were thoroughly reviewed, refactored to pass strict mypy and flake8 checks, and empirically validated via local testing.
* **README:** AI tools were used to design this README.