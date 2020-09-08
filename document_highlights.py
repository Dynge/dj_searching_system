import pandas as pd
import numpy as np
import whoosh
import logging

import data_constants as dc
import searching_module as sm
import preprocessing as pp
import re


def get_highlight(hit, log_level=logging.INFO):
    '''
    '''

    _raw_documents_df = pd.read_csv(dc.DATA_DIR + dc.RAW_DATA_NAME)

    _hi_settings = whoosh.highlight.Highlighter(
        fragmenter=whoosh.highlight.ContextFragmenter(maxchars=200,
                                                      surround=30),
        formatter=whoosh.highlight.HtmlFormatter(between='... \n'))

    _text = _hi_settings.highlight_hit(
        hit,
        'body',
        text=re.sub(r'[\s\t\n]+', ' ',
                    _raw_documents_df.loc[hit['id'], 'Document']),
        top=3,
        minscore=0)
    return (_text)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    raw_documents_df = pd.read_csv(dc.DATA_DIR + dc.RAW_DATA_NAME)

    query = 'forretningsgang bankrådgiver'
    searcher, results = sm.search_over_index(query,
                                             dc.DATA_DIR,
                                             dc.INDEX_DIR,
                                             dc.MODEL_NAME,
                                             limit_results=10)

    first_hit = results[0]
    first_hit_highlight = get_highlight(first_hit)

    logger.info(
        'The following is the snippet from the first document matched: \n"{}"'.
        format(first_hit_highlight))
