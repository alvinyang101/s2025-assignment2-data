import fasttext
import re
from nltk import word_tokenize
from pathlib import Path

QUALITY_MODEL_PATH = Path(__file__).parent.parent / "models" / "wikipedia_quality_model.bin"



def gopher_quality_filter(text):
    """
    Not quality if:
    Contains less than 50 or more than 100,000 words
    Has a mean word length outside the range of 3 to 10 characters
    Has more than 30% of lines ending with an ellipsis ("...")
    Contains less than 80% of words with at least one alphabetic character
    
    Args:
        text (str): The text to validate
        
    Returns:
        bool: True if text passes all constraints, False otherwise
    """

    if not text:
        return False
    
    words = word_tokenize(text)
    
    # Word count
    word_count = len(words)
    if word_count < 50 or word_count > 100000:
        return False
    
    # Mean word length
    word_lengths = [len(word) for word in words if word.isalnum()]
    if not word_lengths:
        return False
    
    mean_word_length = sum(word_lengths) / len(word_lengths)
    if  mean_word_length < 3 or  mean_word_length > 10:
        return False
    
    # Lines ending with ellipsis
    lines = text.split('\n')
    ellipsis_lines = sum(1 for line in lines if line.strip().endswith('...'))
    ellipsis_percentage = ellipsis_lines / len(lines)
    if ellipsis_percentage > 0.3:  # 30%
        return False
    
    # Words with alphabetic characters
    alphabetic_words = sum(1 for word in words if any(c.isalpha() for c in word))
    alphabetic_percentage = alphabetic_words / len(words)
    if alphabetic_percentage < 0.8:
        return False
    
    return True


def classify_quality(text: str) -> tuple[str, float]:
    """
    Scores the quality of the texte
    """
    try:
        model = fasttext.load_model(str(QUALITY_MODEL_PATH))
    except Exception as e:
        raise RuntimeError(f"Failed to load the FastText model: {e}")
        
    text = text.replace('\n', ' ').strip()
    if not text:
        return ("good", 0.0)
    
    # (labels, probabilities)
    predictions = model.predict([text])
    
    label = predictions[0][0][0]  # First label
    confidence = predictions[1][0]  # Confidence
    
    if label.startswith("__label__"):
        label = label[9:]
    
    return label, float(confidence)