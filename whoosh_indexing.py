from whoosh.fields import Schema, TEXT, NUMERIC, KEYWORD, ID, STORED
from whoosh.formats import Frequency
from whoosh.analysis import StemmingAnalyzer
from whoosh import index, writing
import logging
import pandas as pd
import os

import data_constants as dc
import preprocessing as pp

SCHEMA = Schema(  # Schema is the rules and structure (specifications of fields) of the index. 
    id=NUMERIC(
        numtype=int,  # Specifies that it is Ints and not floats (saves memory)
        unique=True,  # Specifies that the field if unique
        signed=False,  # Specifies that there must not be negative 
        stored=
        True  # Specifies that the data should be retrievable after a search. Fields that arent stored can still be searched through.
    ),
    title=TEXT(stored=True, vector=True),
    path=TEXT(stored=False),
    body=TEXT(
        #analyzer=StemmingAnalyzer(), # Automatic Stemmer. Does not do any stemming without
        stored=False,
        vector=True),
    last_update=TEXT(stored=False))


def add_documents_to_index(writer, dataframe, log_level=logging.INFO):
    '''
    Add documents from dataframe to specified index.\n
    The dataframe must contains the following coloumns "Title", "Path", "Document" and "Last Updated".
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    for row in dataframe.index:
        _id_str = int(row)
        _title_str = str(dataframe.loc[row, 'Title'])
        _path_str = str(dataframe.loc[row, 'Path'])
        _document_str = str(dataframe.loc[row, 'Document'])
        _last_updated_str = str(dataframe.loc[row, 'Last Updated'])
        logger.debug('Adding document with title {}'.format(_title_str))

        writer.update_document(id=_id_str,
                               title=_title_str,
                               path=_path_str,
                               body=_document_str,
                               last_update=_last_updated_str)


def populate_index(dirname, dataframe, schema, log_level=logging.INFO):
    '''
    Populates the index based on a dataframe and a schema.\n
    The index is saved in the directory found by "dirname". 
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    if not os.path.exists(dirname):
        os.mkdir(dirname)
        logger.info('Creating and using directory: {}'.format(dirname))
    else:
        logger.info('Using existing directory: {}'.format(dirname))

    ix = index.create_in(
        dirname, schema
    )  # Creates index in the location of dirname with the rules of the schema.
    with ix.writer() as index_writer:
        logger.info('Populating index...')
        add_documents_to_index(index_writer, dataframe)
        index_writer.mergetype = writing.CLEAR


if __name__ == "__main__":
    FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)

    documents_df = pd.read_csv(dc.DATA_DIR + dc.DATA_NAME)
    populate_index(dc.INDEX_DIR, documents_df, SCHEMA)
    logger.info('Finished populating our index.')

    # raw_documents_df = pd.read_csv(dc.DATA_DIR + dc.RAW_DATA_NAME)
    # cleaned_documents_df = raw_documents_df.assign(
    #     Document=lambda df: df['Document'].map(lambda text: pp.clean_data(text)
    #                                            ),
    #     Path=lambda df: df['Path'].map(lambda text: pp.clean_data(text)),
    #     Title=lambda df: df['Title'].map(lambda text: pp.clean_data(text)))

    # populate_index(dc.INDEX_DJ_DIR, cleaned_documents_df, SCHEMA)
    # logger.info('Finished populating Djurslands Bank index.')