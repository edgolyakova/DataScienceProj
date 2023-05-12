import spacy
import stanza

from bs4 import BeautifulSoup
import requests

import os
import re
import math

import numpy as np

nlp = spacy.load('en_core_web_sm')
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos')


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
    Tokenize a string into sentences using Spacy.
    By default the output is a list of strings, however, a list of Spacy sentences can be returned as well.
    """
    doc = nlp(text)
    return list(map(lambda x: str(x), doc.sents)) if to_string else list(doc.sents)


def stanza_get_sents(text, to_string=True):
    """
    Tokenize a string into sentences using Spacy.
    By default the output is a list of strings, however, a list of Stanza sentences can be returned as well.
    """
    doc = nlp_stanza(text)
    return list(map(lambda x: x.text, doc.sentences)) if to_string else doc.sentences


def spacy_tokenize_text(doc, is_pretokenized=False, no_filtering=False, to_string=True, to_lowercase=False):
    """
    Return the list of tokens provided by Spacy.
    The function can both work with a string or a Spacy sentence.
    By default tokens are returned as strings and we expect to tokenize the text within the function.
    """
    if not is_pretokenized:
        doc = nlp(doc) if not to_lowercase else nlp(doc.lower())

    # Tokens can be returned as a list of strings
    if to_string:
        return list(map(lambda x: x.text, doc)) if no_filtering else \
            [token.text for token in doc if not token.is_punct and not token.is_stop]
    # Tokens can be as well return as a list of Spacy tokens
    else:
        return list(doc) if no_filtering else [token for token in doc if not token.is_punct and not token.is_stop]


def stanza_tokenize_text(doc, is_pretokenized=False, to_string=True):
    """
    Return a list of tokens provided by Stanza.
    The function can take a string or a Stanza sentence.
    By default tokens are returned as strings and we expect to tokenize the text within the function.
    """

    if not is_pretokenized:
        doc = nlp_stanza(doc)

    # Tokens can be returned as a list of strings
    if to_string:
        return [x.text for sent in doc.sentences for x in sent.tokens]
    # Or as a list of Stanza tokens
    else:
        return [token for sent in doc.sentences for token in sent.tokens]


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


def get_tokens_dict(tokens):
    # For faster computation and comparison with other tokens we can store them in a hashable dictionary
    tokens_dict = {}

    for t in tokens:
        # Create a dictionary which will have the token text (wordform) as its key and as value it will have
        # an array of all occurences of such wordform
        t_list = tokens_dict.get(t.text, [])
        t_list.append(t)
        tokens_dict[t.text] = t_list

    return tokens_dict


def common_tokens(sp_tokens, st_tokens):
    sp_t_dict = get_tokens_dict(sp_tokens)
    st_t_dict = get_tokens_dict(st_tokens)

    # Find wordforms that are found by both libraries
    same_wf = set(sp_t_dict.keys()).intersection(st_t_dict.keys())

    # Since the same wordform can be in a same twice or more we need to compare all of the wordforms appear
    # equal number of times for both libraries: otherwise one the libraries mistokenized one of the words and
    # we won't know which one. That's why such cases won't be considered as same tokens.

    # Since we can have different number of occurences for the same wordform, we only take the first n occurences
    # where n is the smallest number of occurences between Spacy and Stanza for a selected wordform.
    common_t = list(map(lambda y: (y[0][:min(len(y[0]), len(y[1]))], y[1][:min(len(y[0]), len(y[1]))]),
                        [(sp_t_dict[x], st_t_dict[x]) for x in same_wf]))

    # Return
    return common_t


def get_entities(text):
    """
    We create a function to extract the entities from a text
    """
    list_entities= []
    for ent in nlp(text).ents:
        list_entities.append(ent.label_)
    return(list_entities)