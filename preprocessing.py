from data_constants import DATA_DIR, RAW_DATA_NAME, DATA_NAME
from lemma import lemmatize_documents, lemmatize
from stopword import remove_stopwords_df, remove_stopwords
import pandas as pd
import logging
import re


def clean_data(text, log_level=logging.INFO):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Regex matches dots that arent preceded by a character or number (end of sentence)
    # Regex also matches any non (char or number or dot, comma, question ?, : or !).
    _first_regex = r"(?<![a-zæøå0-9])[.:?!]+ ?|[^a-zæøå0-9.,?!:]+"
    text = re.sub(_first_regex, " ", text.lower())

    # Regex matches ? and ! and :. Converts them to dot. (end of sentence)
    _second_regex = r"[?!:]"
    text = re.sub(_second_regex, ".", text)

    # Regex matches multiple spaces and byte encoding for space to convert to space.
    _third_regex = r" {2,100}| {1}(\xa0)|^ +| +$"
    text = re.sub(_third_regex, " ", text)

    # Regex matches space in the beginning of string and end of string. Replace with nothing.
    _forth_regex = r'^ +| +$'
    text = re.sub(_forth_regex, "", text)
    return (text)


def preprocess_df(documents_df, log_level=logging.INFO):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    _clean_documents_df = documents_df.assign(
        Title=lambda _df: _df['Title'].map(lambda _row: clean_data(_row)),
        Document=lambda _df: _df['Document'].map(lambda _row: clean_data(_row)
                                                 ),
        Path=lambda _df: _df['Path'].map(lambda _row: clean_data(_row)),
    )

    _lemma_documents = lemmatize_documents(_clean_documents_df)

    _lemma_stopword_df = remove_stopwords_df(_lemma_documents)
    return (_lemma_stopword_df)


def preprocess(text, log_level=logging.INFO):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    _clean_text = clean_data(text)
    _lemma_text = lemmatize(_clean_text)
    _lemma_stopword_text = remove_stopwords(_lemma_text)
    return (_lemma_stopword_text)


if __name__ == "__main__":
    FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    documents_df = pd.read_csv(DATA_DIR + RAW_DATA_NAME)

    processed_df = preprocess_df(documents_df)
    processed_df.to_csv(DATA_DIR + DATA_NAME)
