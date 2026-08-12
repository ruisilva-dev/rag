"""Evaluation module for calculating retrieval metrics."""

from src.models import (
    RagDataset,
    StudentSearchResults,
    AnsweredQuestion,
    MinimalSource
)


class SearchEvaluator:
    """Evaluates student search results against ground truth datasets."""

    @staticmethod
    def _is_match(
        correct_src: MinimalSource,
        pred_src: MinimalSource,
        max_chunk_size: int
    ) -> bool:
        """Checks if a predicted source matches the ground truth via IoU >= 5%.

        Args:
            correct_src (MinimalSource): The ground truth source chunk.
            pred_src (MinimalSource): The retrieved student source chunk.
            max_chunk_size (int): Max allowed character length per chunk.

        Returns:
            bool: True if Intersection over Union is at least 5%, else False.
        """
        # File path must match exactly
        if pred_src.file_path != correct_src.file_path:
            return False

        # Ground truth bounds
        correct_start = correct_src.first_character_index
        correct_end = correct_src.last_character_index
        correct_len = correct_end - correct_start

        # Cap student prediction length by max_chunk_size
        pred_start = pred_src.first_character_index
        pred_end = min(
            pred_src.last_character_index,
            pred_start + max_chunk_size
        )
        pred_len = pred_end - pred_start

        # Intersection (Overlap)
        overlap_start = max(correct_start, pred_start)
        overlap_end = min(correct_end, pred_end)
        overlap_len = max(0, overlap_end - overlap_start)

        # Union
        union_len = correct_len + pred_len - overlap_len

        # IoU threshold >= 5%
        return union_len > 0 and (overlap_len / union_len) >= 0.05

    def evaluate(
        self,
        pred_dataset: StudentSearchResults,
        gt_dataset: RagDataset,
        k: int = 10,
        max_chunk_size: int = 2000
    ) -> None:
        """Calculates Recall@k using Intersection over Union (IoU).

        Verifies if retrieved sources have an IoU overlap of at least 5%
        with expected ground truth sources.

        Args:
            pred_dataset (StudentSearchResults): The generated search results.
            gt_dataset (RagDataset): The ground truth dataset.
            k (int): Maximum k limit for evaluation. Defaults to 10.
            max_chunk_size (int): Max allowed character length per chunk.
                Defaults to 2000.
        """
        # Filter only questions that have ground truth sources
        gt_questions = {
            q.question_id: q for q in gt_dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }

        if not gt_questions:
            raise RuntimeError(
                "No answered questions found in the ground truth dataset."
            )

        pred_questions = {
            res.question_id: res.retrieved_sources
            for res in pred_dataset.search_results
        }

        # Prepare metrics arrays
        k_values = [1, 3, 5, 10]
        recalls: dict[int, list[float]] = {k_val: [] for k_val in k_values}

        # Calculate IoU and recall
        for q_id, gt_q in gt_questions.items():
            preds = pred_questions.get(q_id, [])
            total_correct = len(gt_q.sources)

            if total_correct == 0:
                continue

            for k_val in k_values:
                top_k_preds = preds[:k_val]
                number_found = 0

                for correct_src in gt_q.sources:
                    # A source is found if any predicted chunk matches it
                    if any(
                        self._is_match(correct_src, pred, max_chunk_size)
                        for pred in top_k_preds
                    ):
                        number_found += 1

                score = number_found / total_correct
                recalls[k_val].append(score)

        # Print Evaluation Results
        print("Evaluation Results")
        print(f"Questions evaluated: {len(recalls[1])}")

        for k_val in k_values:
            if k_val <= k:
                scores = recalls[k_val]
                avg_recall = sum(scores) / len(scores) if scores else 0.0
                print(f"Recall@{k_val}: {avg_recall:.3f}")
