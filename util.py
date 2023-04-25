import re


def text_clean(text):
    # Remove multiple line breaks and tabs with spaces
    text_cleaned = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', text))

    # If there are 2 or more spaces next to each other, replace with just one space
    text_cleaned = re.sub(r' +', ' ', text_cleaned)

    return text_cleaned
