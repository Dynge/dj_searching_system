# %%
from data_constants import DATA_DIR, MODEL_NAME
from gensim.models import KeyedVectors
from whoosh import index
from whoosh.qparser import QueryParser, MultifieldParser
import os
import logging
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from whoosh_search import parseQuery, search_index
from save_data import read_data_from_file
import embedding_query_expansion as eq1


def extract_forward_index(reader, documents, log_level=logging.INFO):
    '''
    Extracts the forward index of the document.
    The forward index contains information about the terms in the document aswell as the term frequency of the terms in the document.

    Outputs the forward index as a dictionary {term: tf}
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.debug(
        'Extracting forward index of documents: {}.'.format(documents))
    _vectors = {
        _document: reader.vector_as("frequency", _document, "body")
        for _document in documents
    }

    _forward_index = {
        _docnum: pd.concat([
            pd.DataFrame([[term, count]], columns=['term', 'count'])
            for term, count in list(_vector)
        ])
        for _docnum, _vector in _vectors.items()
    }
    logger.debug('Term Counts for Document "{}": Shape {}'.format(
        list(_forward_index.keys())[0],
        _forward_index[list(_forward_index.keys())[0]].shape))

    return (_forward_index)


def extract_freq_freq_tables(f_index, log_level=logging.INFO):
    '''
    Extracts the freq of freq tables for the given forward index. 
    A freq of freq table is a table showing how often different term frequencies occur in the document.
    Used for Good Turing Smoothing.

    Output is a dictionary {document_number: pd.DataFrame([freq_of_tf], index=[tfs])}
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.debug('Counting Frequency of Frequency for Document Vectors')
    _freq_freq_tables = {}
    for _docnum, _f_index in f_index.items():
        _freq_freq_tables[_docnum] = _f_index.groupby(
            'count')['term'].nunique()
        logger.debug('Freq of Freq for Document "{}": Shape {} '.format(
            _docnum, _freq_freq_tables[_docnum].shape))

    return (_freq_freq_tables)


def term_matching_probability(reader,
                              query_list,
                              documents,
                              log_level=logging.DEBUG):
    '''
    Returns the term matching probability of the query given a word and the document. 
    The terms are assumed independant of the query therefore this is the same as the tf_q / N_d.

    Output is a dictionary {document_number: probability_of_query}
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.info(
        'Calculating probability of query "{}" given documents "{}"'.format(
            query_list, documents))
    _forward_index = extract_forward_index(reader, documents)
    _freq_freq_document = extract_freq_freq_tables(_forward_index)
    _probability_dictionary = {}
    for _doc in documents:
        probability_term_in_query = []
        for _q in query_list:
            try:
                _tf = _forward_index[_doc][_forward_index[_doc]['term'] ==
                                           _q]['count'][0]
                logger.debug('"{}" in document with frequency {}.'.format(
                    _q, _tf))
            except LookupError:
                _tf = 0
                logger.debug('"{}" not in document.'.format(_q))

            _probability_of_word = _tf / reader.doc_field_length(_doc, 'body')

            logger.debug('Adding word "{}" with probability of {}.'.format(
                _q, _probability_of_word))
            probability_term_in_query.append(_probability_of_word)

        _probability_dictionary[_doc] = np.prod(probability_term_in_query)

    return (_probability_dictionary)


def semantic_matching_probability(searcher,
                                  model_vectors,
                                  documents,
                                  tokenized_original_query,
                                  Z_documents,
                                  sigmoid_a=10):
    '''
    Returns the semantic matching probability of the query given a word and a document.
    This is a semantic relation between the query and the words within the documents. 

    Output is a dictionary {document_number: {document_word: probability_of_query}}
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(
        'Calculating semantic probability of query "{}" given documents "{}"'.
        format(tokenized_original_query, documents))
    _semantic_matching_probability = {}
    for _docnum in documents:
        logger.debug(
            'Calculating Semantic Probability for Document "{}"'.format(
                _docnum))
        _semantic_matching_probability[_docnum] = {}
        _tf_list = list(searcher.vector_as("frequency", _docnum, "body"))

        _doc_vocab = list(Z_documents[_docnum].keys())
        _doc_vocab_size = len(_doc_vocab)
        _normalized_doc_vectors = model_vectors[_doc_vocab] / np.array(
            np.linalg.norm(model_vectors[_doc_vocab], axis=1)).reshape(
                (_doc_vocab_size, 1))

        _query_len = len(tokenized_original_query)
        _normalized_query_vectors = model_vectors[
            tokenized_original_query] / np.array(
                np.linalg.norm(model_vectors[tokenized_original_query],
                               axis=1)).reshape((_query_len, 1))

        _cosine_simul = eq1.sigmoid(
            _normalized_doc_vectors.dot(_normalized_query_vectors.T))
        logger.debug('Shape of Cosine_matrix: {}'.format(_cosine_simul.shape))
        logger.debug('Corner of Cosine_matrix: \n{}'.format(
            _cosine_simul[:2, :2]))

        _tfs_query = []
        for q_term in tokenized_original_query:
            try:
                _tfs_query.append(_tf_list[_tf_list[:][0] == q_term][1])
            except LookupError:
                _tfs_query.append(0)

        _tfs_query = np.array(_tfs_query).reshape(1, len(_tfs_query))
        _Z_values_document = np.array(list(
            Z_documents[_docnum].values())).reshape(len(_doc_vocab), 1)
        _cosine_tf_weighted = _cosine_simul * _tfs_query
        logger.debug('Shape of Cosine_tf_matrix: {}'.format(
            _cosine_tf_weighted.shape))
        logger.debug('Corner of Cosine_tf_matrix: \n{}'.format(
            _cosine_tf_weighted[:2, :2]))
        _cosine_normalised = _cosine_tf_weighted / _Z_values_document
        logger.debug('Shape of Cosine_norm_matrix: {}'.format(
            _cosine_normalised.shape))
        logger.debug('Corner of Cosine_norm_matrix: \n{}'.format(
            _cosine_normalised[:2, :2]))
        _cosine_products = np.prod(_cosine_normalised, axis=1)
        logger.debug('Shape of Cosine_products: {}'.format(
            _cosine_products.shape))
        logger.debug('Corner of Cosine_products: \n{}'.format(
            _cosine_products[:2]))

        _semantic_matching_probability[_docnum] = {
            _document_word: _cosine_products[_idx]
            for _idx, _document_word in enumerate(_doc_vocab)
        }

    return (_semantic_matching_probability)


def good_turing_smoothing(tfs, freq_freq_table, log_level=logging.INFO):
    '''
    Calculates the Good Turing Smoothing of term frequencies. 

    Output is a dictionary {term: tf_smoothed}
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.debug(
        'Calculating Good Turing Smoothing with tf: "{}" in table: "{}"'.
        format(tfs, freq_freq_table))

    def _func_powerlaw(x, m, c, c0):
        return c0 + x**m * c

    _x = np.array(
        list(freq_freq_table.index) +
        [x for x in range(1, 10) if x not in list(freq_freq_table.index)])
    _y = np.array(
        list(freq_freq_table) +
        [0 for x in range(1, 10) if x not in list(freq_freq_table.index)])

    _fitted_powerlaw = curve_fit(_func_powerlaw,
                                 _x,
                                 _y,
                                 p0=np.asarray([-1, 10**5, 0]),
                                 maxfev=5000)

    _smoothed_count = {}
    for _term, _tf in tfs.items():
        if False not in [
                idx + 1 in freq_freq_table.index for idx in range(_tf + 1)
        ]:
            _n_plus_one = freq_freq_table.loc[_tf + 1]
            _n = freq_freq_table.loc[_tf]
        else:
            _n_plus_one = _func_powerlaw(_tf + 1, _fitted_powerlaw[0][0],
                                         _fitted_powerlaw[0][1],
                                         _fitted_powerlaw[0][2])
            _n = _func_powerlaw(_tf, _fitted_powerlaw[0][0],
                                _fitted_powerlaw[0][1], _fitted_powerlaw[0][2])
        _smoothed_count[_term] = (_tf + 1) * _n_plus_one / _n
    logger.debug('Good Turing Smoothing Result: {}'.format(_smoothed_count))
    return (_smoothed_count)


def mle_probability_word(words,
                         document,
                         index,
                         gt_smooth=True,
                         freq_freq_table=None,
                         log_level=logging.INFO):
    '''
    Calculates the Maximum Likelihood Estimate of the words given documents.
    Defaults to using Good Turing Smoothing with gt_smooth=True.

    Output is a dictionary {term: mle}
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.debug(
        'Calculating the MLE probability of words in "{}"'.format(document))
    _tfs_of_words_in_doc = list(index.searcher().vector_as(
        "frequency", document, "body"))
    _n_words = len(_tfs_of_words_in_doc)
    _tfs = {
        _term: _tf
        for _term, _tf in _tfs_of_words_in_doc if _term in words
    }
    logger.debug(_tfs)

    if gt_smooth:
        _gt_tfs = good_turing_smoothing(_tfs, freq_freq_table[document])
        _gt_tfs.update(
            (_term, _gt_tf / _n_words) for _term, _gt_tf in _gt_tfs.items())
        logger.debug(_gt_tfs)
        return (_gt_tfs)
    else:
        _tfs.update((_term, _tf / _n_words) for _term, _tf in _tfs.items())
        return (_tfs)


def calculate_probability_of_query_erm(index,
                                       model_vectors,
                                       tokenized_query,
                                       rel_documents,
                                       beta_lin_inter=.2,
                                       sigmoid_a=10,
                                       log_level=logging.INFO):
    '''
    Calculates the probability of a query. This uses term and semantic matching probabilities.
    The result is a linear interpolation of these values.
    The result is sorted in descending order in relation to probability.

    Output is a dictionary {document: { term: probability } }
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    with index.reader() as _reader:
        _term_prob = term_matching_probability(_reader, tokenized_query,
                                               rel_documents)
        _document_vocab = {
            _docnum:
            list(extract_forward_index(_reader, [_docnum])[_docnum]['term'])
            for _docnum in rel_documents
        }

    _Z_document = {
        _docnum: eq1.precompute_similarity_sums(model_vectors,
                                                _document_vocab[_docnum])
        for _docnum in _document_vocab
    }

    with index.searcher() as _searcher:
        _sem_prob = semantic_matching_probability(_searcher,
                                                  model_vectors,
                                                  rel_documents,
                                                  tokenized_query,
                                                  _Z_document,
                                                  sigmoid_a=sigmoid_a)

    _sum_prob = {}
    for _document, _vocab in _sem_prob.items():
        _sum_prob[_document] = {
            term: beta_lin_inter * _term_prob[_document] +
            (1 - beta_lin_inter) * value
            for term, value in _sem_prob[_document].items()
        }
        _sum_prob[_document] = {
            term: value
            for term, value in sorted(
                _sum_prob[_document].items(), key=lambda x: x[1], reverse=True)
        }
    return (_sum_prob)


def calculate_probability_of_word_erm(index,
                                      rel_documents,
                                      probability_of_query_erm,
                                      log_level=logging.INFO):
    '''
    Calculates the probability of the words given the documents.

    Output is a dictionary { document_number: probability }
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    _freq_freq_table = extract_freq_freq_tables(
        extract_forward_index(index.reader(), rel_documents))
    _p_word_document = {}

    logger.debug(_freq_freq_table)
    logger.info('Calculating probability of words in {} documents...'.format(
        len(probability_of_query_erm)))
    for _document, _word_dict in probability_of_query_erm.items():
        logger.debug('Calculating probability of words in document: {}'.format(
            _document))
        _p_word_document[_document] = {}
        _p_word_document[_document] = mle_probability_word(
            list(_word_dict.keys()),
            _document,
            index,
            freq_freq_table=_freq_freq_table)

        # for _word in _word_dict.keys():
        #     _p_word_document[_document][_word] = mle_probability_word(
        #         _word, _document, index, freq_freq_table=_freq_freq_table)
    return (_p_word_document)


def erm_language_model(ix,
                       rel_documents,
                       model_vectors,
                       tokenized_query,
                       vocabulary,
                       eq1_model,
                       alpha_lin_inter=.4,
                       beta_lin_inter=.15,
                       sigmoid_a=10,
                       log_level=logging.INFO):
    '''
    Calculates the probability of words given the ERM language model.

    Output is a pandas DataFrame with two columns: 'term' and 'prob'.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    _probability_query_erm = calculate_probability_of_query_erm(
        ix,
        model_vectors,
        tokenized_query,
        rel_documents,
        beta_lin_inter=beta_lin_inter,
        sigmoid_a=sigmoid_a)

    _probability_word_erm = calculate_probability_of_word_erm(
        ix, rel_documents, _probability_query_erm)
    logger.debug(_probability_word_erm)
    _feedback_language_model = {}
    for _word in vocabulary:
        _feedback_language_model[_word] = 0
        for _document, _q_probability in _probability_query_erm.items():
            try:
                _feedback_language_model[_word] += _q_probability[_word] * \
                    _probability_word_erm[_document][_word]
            except LookupError:
                _feedback_language_model[_word] += 0
    _feedback_language_model_sorted = {
        term: probability
        for term, probability in sorted(_feedback_language_model.items(),
                                        key=lambda item: item[0])
    }
    eq1_model_sorted = {
        term: probability
        for term, probability in sorted(eq1_model.items(),
                                        key=lambda item: item[0])
    }

    sanity_check = _feedback_language_model_sorted.keys(
    ) == eq1_model_sorted.keys()
    if not sanity_check:
        logger.warning(
            'Sanity-check. Expecting "True": {}.'.format(sanity_check))
    else:
        logger.info('Sanity-check. Expecting "True": {}.'.format(sanity_check))
    _vocab_model_words = _feedback_language_model_sorted.keys()
    _eq1_model_sorted_probs = np.array(list(eq1_model_sorted.values())) / sum(
        list(eq1_model_sorted.values()))
    _feedback_language_model_probs = np.array(
        list(_feedback_language_model_sorted.values())) / sum(
            list(_feedback_language_model_sorted.values()))

    logger.info('\n\n\nEQ1 Model:\n{}\n\n\n'.format(list({
        term: probability
        for term, probability in sorted(eq1_model.items(),
                                        key=lambda item: item[1], reverse=True)
    })[:100]))
    logger.info('\n\n\ERM Model:\n{}\n\n\n'.format(list({
        term: probability
        for term, probability in sorted(_feedback_language_model.items(),
                                        key=lambda item: item[1], reverse=True)
    })[:100]))

    _erm_language_model = alpha_lin_inter * \
        _eq1_model_sorted_probs + (1-alpha_lin_inter) * \
        _feedback_language_model_probs

    _erm_language_model_sorted = pd.DataFrame({
        'term': list(_vocab_model_words),
        'prob': _erm_language_model
    }).sort_values('prob', ascending=False)

    logger.info('\n\n\Final Model:\n{}\n\n\n'.format(list(_erm_language_model_sorted.iloc[:50, 0])))

    return (_erm_language_model_sorted)


# %%
