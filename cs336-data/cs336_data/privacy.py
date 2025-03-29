import re

def mask_emails(text: str) -> tuple[str, int]:
    """
    Masks email addresses in text and counts how many were found.
    """

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    matches = re.findall(email_pattern, text)

    count = len(matches)
    masked_text = re.sub(email_pattern, "|||EMAIL_ADDRESS|||", text)
    
    return masked_text, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    Masks phone numbers in text and counts how many were found.
    """
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    
    matches = re.findall(phone_pattern, text)
    
    count = len(matches)
    masked_text = re.sub(phone_pattern, "|||PHONE_NUMBER|||", text)
    
    return masked_text, count


def mask_ips(text: str) -> tuple[str, int]:
    """
    Masks IPv4 addresses in text and counts how many were found.
    """
    ipv4_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    matches = re.findall(ipv4_pattern, text)
    
    count = len(matches)
    masked_text = re.sub(ipv4_pattern, "|||IP_ADDRESS|||", text)
    
    return masked_text, count