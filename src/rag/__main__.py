import fire
import bm25s
from pathlib import Path
from rag.models import MinimalSource, RagDataset, StudentSearchResults
from rag.ingestion.ingestion import IngestionPipeline
from rag.ingestion.discoverer import FileDiscoverer
from rag.bm25.indexer import BM25Indexer
from rag.bm25.engine import BM25SearchEngine
from rag.utils import catch_cli_errors

DEFAULT_INDEX_DIR = "data/processed/bm25_index"


class CLI:
    """Command-Line Interface for managing the RAG pipeline.

    Provides commands to build the BM25 index and search datasets.
    """

    @catch_cli_errors
    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        save_dir: str = "data/processed/bm25_index",
        max_chunk_size: int = 2000
    ) -> None:
        """Discovers, chunks, and indexes files from the repository.

        Args:
            repo_path (str): Path to the target source repository.
            save_dir (str): Directory path to save the BM25 index.
            max_chunk_size (int): Maximum character limit per chunk.
        """
        gen = FileDiscoverer(repo_path).discover_files()
        ingestion_pipeline = IngestionPipeline(max_chunk_size)

        sources: list[MinimalSource] = []
        for file_path in gen:
            sources.extend(ingestion_pipeline.process_file(file_path))

        indexer = BM25Indexer()
        retriever = bm25s.BM25()

        corpus = indexer.build_corpus(sources)
        retriever.index(corpus)

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        indexer.save(save_dir, retriever)

    @catch_cli_errors
    def search(
        self,
        query_string: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        """Searches the indexed codebase for a single query string.

        Args:
            query_string (str): The search query to evaluate against the index.
            k (int): Number of relevant sources to retrieve. Defaults to 10.
            save_directory (str): Directory to store search outputs.
            index_dir (str): Directory containing the BM25 index.
        """
        searcher = BM25SearchEngine.load_from_disk(index_dir)

        minimal_results = searcher.search_to_model("1", query_string, k)
        final_results = StudentSearchResults(
            search_results=[minimal_results], k=k
        )

        raw_json = final_results.model_dump_json(by_alias=True, indent=4)

        output_file = Path(save_directory) / "single_search_public.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(raw_json)

    @catch_cli_errors
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = DEFAULT_INDEX_DIR
    ) -> None:
        """Processes a dataset of questions and saves search results.

        Args:
            dataset_path (str): Path to the JSON dataset.
            k (int): Number of relevant sources to retrieve. Defaults to 10.
            save_directory (str): Directory to store search outputs.
            index_dir (str): Directory containing the BM25 index.
        """
        searcher = BM25SearchEngine.load_from_disk(index_dir)

        dataset_content: str = Path(dataset_path).read_text(encoding="utf-8")
        dataset = RagDataset.model_validate_json(dataset_content)

        results = searcher.batch_search(dataset.rag_questions, limit=k)

        raw_json = results.model_dump_json(by_alias=True, indent=4)

        output_file = Path(save_directory) / Path(dataset_path).name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(raw_json)


if __name__ == "__main__":
    fire.Fire(CLI)
