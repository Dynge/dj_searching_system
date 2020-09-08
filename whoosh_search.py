import whoosh
import logging
import os
import pandas as pd


def parseQuery(schema, fields, text, log_level=logging.INFO):
    '''
    Takes a text and parses the query in relation to specific fields of the index schema.

    Returns a tuple of the parsed query.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.info('Parsing the query on the fields "{}".'.format(fields))
    _qp = whoosh.qparser.MultifieldParser(
        fieldnames=fields,  # Tells the query parser which field to search in
        # Tells the query parser which schema the index follows.
        schema=schema)

    _q = _qp.parse(text)  # Parsing the query
    logger.debug('Parsed query into: {}'.format(_q))

    return (_q)


def search_index(searcher,
                 parsed_query,
                 max_results=10,
                 log_level=logging.INFO):
    '''
    Searches the index for the parsed query. 

    You can set the amount of results to be returned (default to 10).

    Return a Results object containing the top scored documents.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.info('Searching over index with BM25F.')
    logger.debug('Searching with the query: "{}".'.format(parsed_query))
    _results = searcher.search(parsed_query, terms=True, limit=max_results)
    logger.debug('Search results: {}'.format(_results))
    return (_results)
