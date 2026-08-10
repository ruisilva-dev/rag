"""Data models for source chunks, questions, and search results."""

from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """Location and character range metadata for a source chunk."""

    file_path: str
    first_character_index: int
    last_character_index: int
    context_headers: list[str] = []


class UnansweredQuestion(BaseModel):
    """Incoming query without ground truth sources or answers."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Query paired with ground truth sources and expected answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset holding a batch of questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Stores retrieved sources matching a single query."""

    question_id: str
    question: str = Field(serialization_alias="question_str")
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Extends search results to include an LLM-generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Collection wrapper for batch search results."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Collection wrapper for batch search results and answers."""

    search_results: list[MinimalAnswer]
    k: int
