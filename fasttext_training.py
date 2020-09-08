from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
import logging
import os
import pandas as pd

from data_constants import DATA_DIR, DATA_NAME, MODEL_NAME


def train_fasttext(tokenized_docs, size, window, min_n, max_n, min_count,
                   epochs):
    '''
    Trains a FastText model.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    _size = size  # Size of embeddings vector (neurons in the hidden layer)
    _window = window  # Window size. Context words = 2*window
    _min_n = min_n
    _max_n = max_n
    _min_count = min_count  #Words must appear atleast min_count in the corpus to be considered
    _epochs = epochs

    logger.info(
        '\nTraining a FastText model with the following parameters:\nSize = {}\nWindow = {}\nMin_count = {}\nEpochs = {}'
        .format(_size, _window, _min_count, _epochs))

    model = FastText(size=_size,
                     window=_window,
                     min_count=_min_count,
                     min_n=_min_n,
                     max_n=_max_n)
    model.build_vocab(sentences=tokenized_docs)
    model.train(sentences=tokenized_docs,
                total_examples=len(tokenized_docs),
                epochs=_epochs)

    return (model)


def tokenize_documents(document_vector):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.info('Tokenizing the documents')
    tokenized_docs = [str(document).split() for document in document_vector]
    logger.debug('Simple preprocessing of documents: \n{}'.format(
        tokenized_docs[0]))
    return (tokenized_docs)


if __name__ == "__main__":
    FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    documents_df = pd.read_csv(DATA_DIR + DATA_NAME)

    model = train_fasttext(tokenized_docs=tokenize_documents(
        documents_df['Document']),
                           size=60,
                           window=5,
                           min_n=3,
                           max_n=6,
                           min_count=5,
                           epochs=5)
    model.init_sims(replace=False)  # L2-norm the embedding vectors
    model.save(DATA_DIR + MODEL_NAME)
