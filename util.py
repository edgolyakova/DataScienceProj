import spacy

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import requests

import os
import re
import math

nlp = spacy.load('en_core_web_sm')


def text_clean(text):

    substitution_rules = [
        # Remove multiple line breaks and replace with a space
        (r'\n+', ' '),
        # Remove multiple tabs and replace with a singular space
        (r'\t+', ' '),
        # Completely remove special characters like ^, <, >, |
        (r'[\|\>\<\^\+=]', ''),
        # Remove multiple spaces and replace with a singular space
        (r' +', ' ')
    ]

    for rule in substitution_rules:
        text = re.sub(rule[0], rule[1], text)


    return text


def get_webpage_text(url):
    """
    Request the page and create a BeautifulSoup object to extract all the texts from it
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, features='html.parser')

    # Extract all the text of the page
    text = soup.text

    # Remove extra line breaks and tabulation before returning
    return text_clean(text)


def spacy_get_sents(text, to_string=True):
    """
    Tokenize a string into sentences.
    By default the output is a list of strings, however, a list of Spacy sentences can be returned as well.
    """
    doc = nlp(text)
    return list(map(lambda x: str(x), doc.sents)) if to_string else doc.sents


def spacy_tokenize_text(text, is_pretokenized=False):
    """
    Return the list of tokens.
    The function can both work with a string or a Spacy sentence.
    """
    if not is_pretokenized:
        text = nlp(text)

    return [token.text for token in text if not token.is_punct and not token.is_stop and not token.is_space]


def get_texts(source_type='folder', path='', token_type='file', n=None):
    """
    The functiton allows to either extract text files from an existing folder or request a new page using a URL.
    Accepted source_type = folder|url
    path needs to be a destination URL if source_type = 'url' and folder path if source_type = 'folder'.
    The function also allows to specify whether the whole file needs to be returned or n tokens or n sentences.
    Accepted token_type = file|sentence|token
    If n is set to None, the function will return all tokens of the available format, otherwise it will stop at nth
    element.
    Spacy is used for sentence and token-level tokenization.
    """
    # For any type of token type, the response will be a list of n of tokens (either file, sentence or token)

    # If n is not defined the function will work till we're out of files in a folder
    if not n:
        n = math.inf

    if source_type == 'folder':
        files_in_folder = os.listdir(path=path)
        result = []

        if token_type == 'file':
            # If n was not specified, all files are requested

            # Read all the files and create a string for each file and append it to the result array
            for file in files_in_folder[:min(n, len(files_in_folder))]:
                with open(f'{path}/{file}', 'r') as f:
                    result.append(f.read())

            return result

        else:

            i = 0

            # Loop until we reach the desired length n or out of files in the folder.
            while len(result) < n and i < len(files_in_folder):
                with open(f'{path}/{files_in_folder[i]}', 'r') as f:
                    text = f.read()
                    if token_type == 'sentence':
                        # Tokenize sentences as strings
                        sents = spacy_get_sents(text)
                        result.extend(sents)
                    elif token_type == 'token':
                        # Separate the text into individual tokens
                        tokens = spacy_tokenize_text(text)
                        result.extend(tokens)
                    else:
                        # If any other token_type was passed raise an error
                        raise Exception('Invalid token_type. Only accepted values are: text, sentence, token.')
                # Increase the index to move to the next file in folder
                i += 1

            # Return all results id n was not set originally, or return n results if it was requested.
            return result[:min(n, len(result))]

    elif source_type == 'url':
        text = get_webpage_text(path)

        if token_type == 'file':
            # We ignore n completely for the source type URL and tokenization type "file" since there is just one file
            return [text]
        elif token_type == 'sentence':
            sents = spacy_get_sents(text)
            # Return either all available sentences if n is more than the total number of sentences
            # or was not defined. Otherwise, return n sentences.
            return sents[:min(n, len(sents))]
        elif token_type == 'token':
            tokens = spacy_tokenize_text(text)
            # Return either all available sentences if n is more than the total number of tokens
            # or was not defined. Otherwise, return n tokens.
            return tokens[:min(n, len(tokens))]
