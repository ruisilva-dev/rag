import re
import Stemmer
import bm25s

ENGLISH_STEMMER = Stemmer.Stemmer("english")
ENGLISH_STOPWORDS = set(bm25s.stopwords.STOPWORDS_EN_PLUS)


def tokenize(text: str, is_code: bool) -> list[str]:
    """Normalizes text to lowercase and extracts alphanumeric tokens.

    Args:
        text (str): The raw text string to tokenize.

    Returns:
        list[str]: A list of clean word tokens.
    """
    if is_code:
        text = text.replace('_', ' ')
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        return re.findall(r'\w+', text.lower())
    else:
        words: list[str] = re.findall(r'\w+', text.lower())
        filtered_words: list[str] = [
            word for word in words if word not in ENGLISH_STOPWORDS
        ]
        stemmed_words = ENGLISH_STEMMER.stemWords(filtered_words)
        return stemmed_words
