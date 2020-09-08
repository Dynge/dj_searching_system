import stanza
import pandas as pd
import numpy as np
import logging
import os
import re
import time

from data_constants import DATA_DIR, RAW_DATA_NAME, LEMMA_NAME


def lemmatize(text, nlp=None):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.debug('Lemmatizing the text: "{}"'.format(text))
    if nlp == None:
        logger.warning(
            'Running lemmatize without inputting an nlp pipeline is slow if repeated.'
        )

        stanza.download('da')
        nlp = stanza.Pipeline('da',
                              processors='tokenize, pos, lemma',
                              verbose=False)
    if re.match(r'[a-zæøå]', text):
        _lemma_text = [
            word.lemma for sentence in nlp(text).sentences
            for word in sentence.words
        ]

        return (" ".join(_lemma_text))
    return (text)


def lemmatize_documents(documents):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    _sample_size = len(documents.index)

    stanza.download('da')
    _nlp = stanza.Pipeline('da',
                           processors='tokenize, pos, lemma',
                           verbose=False)

    def _count_lemmatize(row, text, nlp, total_len="", report_every=10):
        logger.debug('Lemmatizing row {} of {}.'.format(row, total_len))
        if row % report_every == 0:
            logger.info('Lemmatizing row {} of {}.'.format(row, total_len))
        return (lemmatize(text, nlp))

    _doc_lem = [
        _count_lemmatize(row, text, _nlp, _sample_size)
        for row, text in enumerate(documents['Document'])
    ]
    _title_lem = [lemmatize(text, _nlp) for text in documents['Title']]
    _path_lem = [lemmatize(text, _nlp) for text in documents['Path']]

    _lemmatized_df = documents.iloc[0:_sample_size].assign(Title=_title_lem,
                                                           Document=_doc_lem,
                                                           Path=_path_lem)
    return (_lemmatized_df)


if __name__ == "__main__":
    FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    documents_df = pd.read_csv(DATA_DIR + RAW_DATA_NAME)

    lemmatized_df = lemmatize_documents(documents_df)
    logger.info('Saving lemmatized dataframe to csv: {}'.format(DATA_DIR +
                                                                LEMMA_NAME))
    lemmatized_df.to_csv(DATA_DIR + LEMMA_NAME)
