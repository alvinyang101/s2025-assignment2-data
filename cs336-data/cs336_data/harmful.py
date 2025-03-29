import fasttext
from pathlib import Path

NSFW_MODEL_PATH = Path(__file__).parent.parent / "models" / "jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_MODEL_PATH = Path(__file__).parent.parent / "models" / "jigsaw_fasttext_bigrams_hatespeech_final.bin"


def classify_nsfw(text: str) -> tuple[str, float]:
    try:
        model = fasttext.load_model(str(NSFW_MODEL_PATH))
    except Exception as e:
        raise RuntimeError(f"Failed to load the FastText model: {e}")
        
    if not text:
        return ("non-nsfw", 0.0)
    text = text.replace('\n', ' ').strip()
    # (labels, probabilities)
    predictions = model.predict([text])
    
    label = predictions[0][0][0]  # First label
    confidence = predictions[1][0]  # Confidence
    
    if label.startswith("__label__"):
        label = label[9:]
    
    return label, float(confidence)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    try:
        model = fasttext.load_model(str(TOXIC_MODEL_PATH))
    except Exception as e:
        raise RuntimeError(f"Failed to load the FastText model: {e}")
        
    if not text:
        return ("non-toxic", 0.0)
    text = text.replace('\n', ' ').strip()
    # (labels, probabilities)
    predictions = model.predict([text])
    
    label = predictions[0][0][0]  # First label
    confidence = predictions[1][0]  # Confidence
    
    if label.startswith("__label__"):
        label = label[9:]
    
    return label, float(confidence)
