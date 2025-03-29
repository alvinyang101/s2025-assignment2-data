import resiliparse.parse.encoding
from resiliparse.extract.html2text import extract_plain_text
from typing import Optional

def extract_text_from_html_bytes(html_bytes: bytes) -> Optional[str]:
    """    
    Args:
        html_bytes (bytes)
        
    Returns:
        A string containing the extracted text, or None if extraction fails
    """
    try:
        detected_encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
        try:
            html_str = html_bytes.decode(detected_encoding)
        except UnicodeDecodeError:
            html_str = html_bytes.decode('utf-8', errors='replace')
            
        extracted_text = extract_plain_text(html_str)
        return extracted_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None