"""LLM Answer Generation module."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

SYSTEM_PROMPT = (
    "You are an expert coding assistant. You must answer the user's "
    "question using ONLY the provided context. Do not hallucinate. "
    "When asked about API endpoints, commands, or parameters, always "
    "extract the exact technical string or path rather than describing its "
    "section. "
    "Keep your answer concise, direct, and self-contained. "
    "MANDATORY RULE: At the very end of your answer, you must explicitly "
    "cite the source file path exactly as provided in the context "
    "(for instance, 'Source: data/raw/...')."
)


class AnswerGenerator:
    """Generates source-grounded answers using causal language models."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initializes the tokenizer, model, and generation pipeline.

        Args:
            model_name (str): Hugging Face model repository identifier.
                Defaults to "Qwen/Qwen3-0.6B".

        Raises:
            RuntimeError: If model initialization or device assignment
                fails.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
            ).to(self.device)  # type: ignore

            self.generator = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}': {e}"
            ) from e

    def generate(self, question: str, context: str) -> str:
        """Generates an answer to a question using provided context.

        Args:
            question (str): The user query to answer.
            context (str): Formatted source text context.

        Returns:
            str: Generated answer containing source citation or error
                message.
        """
        user_content = (
            f"Context:\n---\n{context}\n---\n\n"
            f"Question: {question}\n\n"
            "Remember: Answer the question using ONLY the context above. "
            "At the very end of your response, you must cite the "
            "source file path."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        try:
            outputs = self.generator(
                str(prompt),
                max_new_tokens=512,
                clean_up_tokenization_spaces=False,
                return_full_text=False,
                do_sample=False
            )
            text = str(outputs[0]["generated_text"]).strip()
        except Exception as e:
            return f"Error: Failed to generate an answer: {e}"

        return text
