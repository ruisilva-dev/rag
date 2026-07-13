import re


def tokenize(text: str) -> list[str]:
    """Normalizes text to lowercase and extracts alphanumeric tokens.

    Args:
        text (str): The raw text string to tokenize.

    Returns:
        list[str]: A list of clean word tokens.
    """
    return re.findall(r'\w+', text.lower())
