import re

def split_text_by_keywords(text, keywords):
    # Create a regex pattern to match the keywords
    pattern = r'(?=^' + '|^'.join(map(re.escape, keywords)) + r')'

    # Split the text using the pattern
    sections = re.split(pattern, text, flags=re.MULTILINE)

    # Filter out any empty strings in the result
    sections = [section.strip() for section in sections if section.strip()]

    return sections