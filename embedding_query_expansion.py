
from functools import reduce
import pandas as pd
import numpy as np
import operator
import logging
import time


def sigmoid(x, a = 10):
    '''
    Calculates the sigmoid of X.
    '''
    c = .8
    return(1 / (1 + np.exp(-a * (x-c))))

def sigmoid_cosine(model_vectors, word1, word2, sigmoid_a = 10):
    '''
    Uses the model_vectors from the fasttext model to calculate the cosine similarity between word1 and word2.
    These cosine similarities are then passed to a sigmoid function in order punish cosine similarity harder.
    '''
    sigmoid_cos = sigmoid(x=model_vectors.similarity(word1, word2), a=sigmoid_a)
    return(sigmoid_cos)


def precompute_similarity_sums(model_vectors, vocab, a_sigmoid=10, log_level = logging.INFO):
    '''
    Takes a FastText model_vector object and a vocabulary.
    Output is a dictionary containing the sums of the term similarity to the other terms in the vocabulary.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    _embedding_size = len(model_vectors[vocab[0]])
    _vocab_size = len(vocab)
    logger.debug('Size of word vector array for the vocabulary: {}'.format(
        model_vectors[vocab].shape))

    _normalized_model_vectors = model_vectors[vocab] / np.array(
        np.linalg.norm(model_vectors[vocab], axis=1)).reshape((_vocab_size, 1))
    logger.debug('Size of normalized word vector array: {}'.format(
        _normalized_model_vectors.shape))

    _cosine_simul = sigmoid(
        _normalized_model_vectors.dot(_normalized_model_vectors.T), a=a_sigmoid)
    logger.debug('Size of cosine similarity word array: {}'.format(
        _cosine_simul.shape))
    logger.debug('Sample of cosine similarity: \n{}'.format(
        _cosine_simul[:2,:2]))
    _cosine_sum = np.sum(_cosine_simul, axis=1)
    logger.debug('Size of cosine sum word array: {}'.format(
        _cosine_sum.shape))
    logger.debug('Sample of cosine sum: \n{}'.format(
        _cosine_sum[:2]))

    _similarity_sums = {_term: _sum for _term, _sum in zip(vocab, _cosine_sum)}
    
    return(_similarity_sums)


def compute_word_query_similarity(model_vectors, query_list, vocabulary, p_word, sigmoid_a = 10, log_level = logging.INFO):
    '''
    Computes the probability of the query_list given the words in the vocabulary.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    _p_q_given_word = {}
    for _vocab_word in vocabulary:
        _p_q_given_word.update({_vocab_word: []})
        logger.debug(
            'Calculating probability for query given "{}"'.format(_vocab_word))
        for _query_word in query_list:
            _p_q_given_word[_vocab_word].append(sigmoid_cosine(
                model_vectors, _vocab_word, _query_word, sigmoid_a=sigmoid_a) / p_word[_vocab_word])
    return(_p_q_given_word)


def prod(iterable):
    '''
    Computes the product of the items in the iterable.
    '''
    return reduce(operator.mul, iterable, 1)


def compute_probability_word_expansion(model_vectors, vocabulary, p_word, query_list, sigmoid_a = 10, log_level = logging.INFO):
    '''
    Computes the probability of the words in the vocabulary given the query_list. 
    Returns a sorted dictionary of the words and their respective probability to cooccur with the query.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.info('Computing probabilities for word expansions...')
    _p_q_given_word = compute_word_query_similarity(
        model_vectors, query_list, vocabulary, p_word, sigmoid_a=sigmoid_a)
    _p_word_given_querymodel = {}
    for _vocab_word in vocabulary:
        logger.debug('_p_word_given_querymodel for {}'.format(_vocab_word))
        _p_word_given_querymodel.update(
            {_vocab_word: p_word[_vocab_word] * prod(_p_q_given_word[_vocab_word])})
    _normalize_prob = sum(list(_p_word_given_querymodel.values()))
    _sorted_words = {_vocab_word: _probability / _normalize_prob for _vocab_word, _probability in sorted(
        _p_word_given_querymodel.items(), key=lambda _item: _item[1], reverse=True)}
    return(_sorted_words)
