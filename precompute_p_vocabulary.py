import gensim.models
from functools import reduce
import logging
import pandas as pd

import data_constants as dc
import embedding_query_expansion as eq1
import save_data
import fasttext_training


def get_mle_word(listOfElements, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    indexPosList = []
    indexPos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            indexPos = listOfElements.index(element, indexPos)
            # Add the index position in list
            indexPosList.append(indexPos)
            indexPos += 1
        except ValueError as e:
            logger.debug('Value not in index anymore')
            break

    return len(indexPosList) / float(len(listOfElements))


if __name__ == "__main__":
    FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    model = gensim.models.KeyedVectors.load(dc.DATA_DIR + dc.MODEL_NAME,
                                            mmap='r')

    model_vectors = model.wv
    model_vocab = list(model_vectors.vocab)

    a_sigmoid_range = [10]
    for a_sigmoid in a_sigmoid_range:
        p_word = eq1.precompute_similarity_sums(model_vectors,
                                                model_vocab,
                                                a_sigmoid=a_sigmoid,
                                                log_level=logging.INFO)
        save_data.save_data_to_file(
            p_word, dc.DATA_DIR + dc.get_p_vocab_name(a_sigmoid=a_sigmoid))

    # document_df = pd.read_csv(dc.DATA_DIR + dc.DATA_NAME)

    # document_tokens = reduce(
    #     lambda x, y: x + y,
    #     fasttext_training.tokenize_documents(document_df['Document']))
    # mle_word_in_vocab = {
    #     word: get_mle_word(document_tokens, word)
    #     for word in model_vocab
    # }

    # save_data.save_data_to_file(mle_word_in_vocab, dc.DATA_DIR + dc.MLE_WORDS)
