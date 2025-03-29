import fasttext
from pathlib import Path
from typing import Any

MODEL_PATH = Path(__file__).parent.parent / "models" / "lid.176.bin"

def identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(str(MODEL_PATH))
    
    text = text.replace('\n', ' ').strip()

    if not text:
        return ("en", 0.0)

    try:
        predictions = model.predict([text], k=1)
        language = predictions[0][0][0].replace('__label__', '')
        confidence = float(predictions[1][0])
    except ValueError as e:
        print(f"ValueError processing text: {e}")
        return "en", 0.0
    
    return language, confidence
