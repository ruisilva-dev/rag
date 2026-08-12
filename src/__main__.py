"""Command-line interface entry point for the RAG pipeline."""

import fire
import bm25s
from pathlib import Path
from tqdm import tqdm
from src.models import (
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from src.ingestion import FileDiscoverer, FileChunker
from src.bm25.indexer import BM25Indexer
from src.bm25.engine import BM25SearchEngine
from src.utils import catch_cli_errors
from src.generation.context import ContextBuilder
from src.generation.generator import AnswerGenerator
from src.evaluation import SearchEvaluator

DEFAULT_INDEX_DIR = "data/processed/bm25_index"


class CLI:
    """Command-Line Interface for managing the RAG pipeline.

    Provides commands to index repositories, search datasets, and
    generate source-grounded answers.
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
        # Discover target files
        gen = FileDiscoverer(repo_path).discover_files()

        # Extract and chunk text
        sources: list[MinimalSource] = []
        for file_path in tqdm(gen, desc="Indexing files"):
            sources.extend(FileChunker(max_chunk_size).process_file(file_path))

        indexer = BM25Indexer()
        retriever = bm25s.BM25()

        # Build and train BM25 index
        corpus = indexer.build_corpus(sources)
        retriever.index(corpus)

        # Save to disk
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
        # Load the BM25 index
        searcher = BM25SearchEngine.load_from_disk(index_dir)

        # Perform the search
        minimal_results = searcher.search_to_model("1", query_string, k)
        final_results = StudentSearchResults(
            search_results=[minimal_results], k=k
        )

        # Package and save results
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
        # Load the BM25 index
        searcher = BM25SearchEngine.load_from_disk(index_dir)

        # Load and validate the dataset
        dataset_content: str = Path(dataset_path).read_text(encoding="utf-8")
        dataset = RagDataset.model_validate_json(dataset_content)

        # Perform batch search
        results = searcher.batch_search(dataset.rag_questions, limit=k)

        # Package and save results
        raw_json = results.model_dump_json(by_alias=True, indent=4)

        output_file = Path(save_directory) / Path(dataset_path).name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(raw_json)

    @catch_cli_errors
    def answer(
        self,
        query_string: str,
        k: int = 10,
        save_directory: str = "data/output/answer_results",
        index_dir: str = DEFAULT_INDEX_DIR
    ) -> None:
        """Generates an answer for a single query using retrieved context.

        Args:
            query_string (str): The specific query to answer.
            k (int): Number of relevant sources to retrieve. Defaults to 10.
            save_directory (str): Directory to store the output JSON.
            index_dir (str): Directory containing the BM25 index.
        """
        # Retrieve relevant sources
        searcher = BM25SearchEngine.load_from_disk(index_dir)

        minimal_search = searcher.search_to_model("1", query_string, k)
        search_results = StudentSearchResults(
            search_results=[minimal_search], k=k
        )
        sources = search_results.search_results[0].retrieved_sources

        # Reconstruct context from source files
        if k == 0 or not sources:
            answer_str = (
                "Error: No context retrieved (k=0 or empty index). "
                "Cannot generate answer."
            )
        else:
            context: str = ContextBuilder().build(sources)
            # Generate answer using LLM
            answer_str = AnswerGenerator().generate(query_string, context)

        minimal_answer = MinimalAnswer(
            **minimal_search.model_dump(),
            answer=answer_str
        )

        # Output and save results
        answer_results = StudentSearchResultsAndAnswer(
            search_results=[minimal_answer],
            k=k
        )

        print(answer_str)

        raw_json = answer_results.model_dump_json(by_alias=True, indent=4)

        output_file = Path(save_directory) / "single_answer.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(raw_json)

    @catch_cli_errors
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        """Generates answers for a dataset of search results.

        Args:
            student_search_results_path (str): Path to the JSON file containing
                previously retrieved search results.
            save_directory (str): Directory to save the final answers JSON.
        """
        # Load the search results
        dataset_content = Path(student_search_results_path).read_text(
            encoding="utf-8"
        )
        search_dataset = StudentSearchResults.model_validate_json(
            dataset_content
        )

        context_builder = ContextBuilder()
        answer_generator = AnswerGenerator()
        answered_results = []

        print(
            f"Loaded {len(search_dataset.search_results)} questions from "
            f"{student_search_results_path}"
        )

        # Loop through with tqdm progress bar
        for result in tqdm(
            search_dataset.search_results, desc="Processing questions"
        ):
            if search_dataset.k == 0 or not result.retrieved_sources:
                answer_str = (
                    "Error: No context retrieved. "
                    "Cannot generate answer."
                )
            else:
                context = context_builder.build(result.retrieved_sources)
                answer_str = answer_generator.generate(
                    result.question, context
                )

            minimal_answer = MinimalAnswer(
                **result.model_dump(),
                answer=answer_str
            )
            answered_results.append(minimal_answer)

        # Package and save
        final_dataset = StudentSearchResultsAndAnswer(
            search_results=answered_results,
            k=search_dataset.k
        )

        raw_json = final_dataset.model_dump_json(by_alias=True, indent=4)
        output_file = (
            Path(save_directory) / Path(student_search_results_path).name
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(raw_json)

        print(f"Saved student_search_results_and_answer to {output_file}")

    @catch_cli_errors
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
        max_chunk_size: int = 2000
    ) -> None:
        """Evaluates search results against ground truth.

        Calculates Recall@k by verifying if retrieved sources have an
        Intersection over Union (IoU) overlap of at least 5% with expected
        ground truth sources.

        Args:
            student_search_results_path (str): Path to generated JSON answers.
            dataset_path (str): Path to ground truth dataset JSON.
            k (int): Maximum k limit for evaluation. Defaults to 10.
            max_chunk_size (int): Max allowed character length per chunk.
                Defaults to 2000.
        """
        # Load search results
        pred_content = Path(student_search_results_path).read_text(
            encoding="utf-8"
        )
        pred_dataset = StudentSearchResults.model_validate_json(pred_content)

        # Load Ground Truth Dataset
        gt_content = Path(dataset_path).read_text(encoding="utf-8")
        gt_dataset = RagDataset.model_validate_json(gt_content)

        # Execute evaluation logic
        evaluator = SearchEvaluator()
        evaluator.evaluate(
            pred_dataset=pred_dataset,
            gt_dataset=gt_dataset,
            k=k,
            max_chunk_size=max_chunk_size
        )


if __name__ == "__main__":
    fire.Fire(CLI)
