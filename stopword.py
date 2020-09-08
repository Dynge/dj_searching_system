import re
from data_constants import DATA_DIR, LEMMA_NAME, STOP_NAME
import nltk
from nltk.corpus import stopwords
import os
import pandas as pd
import logging


def remove_stopwords(text, log_level=logging.INFO):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    nltk.download('stopwords')
    _da_stopwords = set(stopwords.words('danish'))

    # Matches stopwords if they are preceded by a non-character and succeded by a non-character
    _stopword_regex = r'(?<![a-zæøå_])(' + '|'.join(
        _da_stopwords) + r')(?![a-zæøå_]) ?'
    _remove_dots_regex = r'[.,](?=[^a-zæøå0-9])|[.,]$'
    _combine_regex = '|'.join([_stopword_regex, _remove_dots_regex])
    _text_stop = re.sub(_combine_regex, " ", text)
    _text_stop = re.sub(r' +', ' ', _text_stop)
    return (_text_stop)


def remove_stopwords_df(pandas_df, log_level=logging.INFO):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.info(
        'Removing stopwords from columns Title, Path and Document with size: {}'
        .format(len(pandas_df.index)))
    nltk.download('stopwords')
    _da_stopwords = set(stopwords.words('danish'))

    _stop_removed = pandas_df.assign(
        Document=lambda _df: _df['Document'].map(lambda _text:
                                                 remove_stopwords(_text)),
        Title=lambda _df: _df['Title'].map(lambda _text: remove_stopwords(_text
                                                                          )),
        Path=lambda _df: _df['Path'].map(lambda _text: remove_stopwords(_text)
                                         ))
    return (_stop_removed)


if __name__ == "__main__":
    FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)

    documents_df = pd.read_csv(DATA_DIR + LEMMA_NAME)
    stop_removed = remove_stopwords_df(documents_df)
    stop_removed.to_csv(DATA_DIR + STOP_NAME)
