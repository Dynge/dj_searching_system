import gensim.models
import whoosh
import whoosh.scoring
import pandas as pd

import data_constants as dc
import save_data
import embedding_query_expansion as eq1
import embedding_relevance_model as erm
import whoosh_search as wh_search
import preprocessing as pp
import document_highlights as dh

import logging


def search_over_index(query_text,
                      DATA_DIR,
                      INDEX_DIR,
                      MODEL_NAME,
                      fields=['title', 'body', 'path'],
                      limit_results=10,
                      b_bm25=.75,
                      k1_bm25=1.2,
                      m_eq1=100,
                      alpha_erm=.5,
                      beta_erm=.1,
                      m_erm=50,
                      a_sigmoid=10):
    '''
    Returns a tuple of the (index_searcher, hits).
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    _processed_query = pp.preprocess(query_text)
    logger.info('Opening index in directory "{}".'.format(INDEX_DIR))
    # Opens an already saved index from the directory
    _ix = whoosh.index.open_dir(INDEX_DIR)
    _model = gensim.models.KeyedVectors.load(DATA_DIR + MODEL_NAME, mmap='r')
    _model_vectors = _model.wv

    _tokenized_original_query = _processed_query.split()

    # Previous Word Expansion EQ1
    _model_vocab = list(_model_vectors.vocab)
    _sample_size = int(len(_model_vocab))
    _sample_vocab = _model_vocab[:_sample_size]

    _p_word = save_data.read_data_from_file(DATA_DIR + dc.get_p_vocab_name(
        a_sigmoid=a_sigmoid))

    _eq1_model = eq1.compute_probability_word_expansion(
        _model_vectors,
        _sample_vocab,
        _p_word,
        _tokenized_original_query,
        sigmoid_a=a_sigmoid)
    _top_n_terms_eq1 = [(term, _eq1_model[term])
                        for term in list(_eq1_model.keys())[:m_eq1]]
    _expanded_query = " OR ".join(_tokenized_original_query +
                                  [term for term, prob in _top_n_terms_eq1])
    _q = wh_search.parseQuery(_ix.schema, fields, _expanded_query)
    _token_expand_query = list(set([term for field, term in _q.all_terms()]))
    with _ix.searcher(
            weighting=whoosh.scoring.BM25F(B=b_bm25, K1=k1_bm25)
    ) as s:  # The searching object and specifying that the custom scorer is used. (Default BM25f)
        _results_eq1 = wh_search.search_index(s, _q, max_results=10)

        _rel_documents = [number for number, score in _results_eq1.items()]

    _erm_model = erm.erm_language_model(_ix,
                                        _rel_documents,
                                        _model_vectors,
                                        _tokenized_original_query,
                                        _sample_vocab,
                                        _eq1_model,
                                        alpha_lin_inter=alpha_erm,
                                        beta_lin_inter=beta_erm,
                                        sigmoid_a=a_sigmoid)

    _top_n_terms_erm = _erm_model.iloc[:m_erm]
    _feedback_tokenized_query = _tokenized_original_query + \
        list(_top_n_terms_erm['term'])
    _q_erm = wh_search.parseQuery(_ix.schema, fields,
                                  " OR ".join(_feedback_tokenized_query))

    _searcher = _ix.searcher(
    )  # The searching object and specifying that the custom scorer is used. (Default BM25f)
    _results_erm = wh_search.search_index(_searcher,
                                          _q_erm,
                                          max_results=limit_results)

    return (_searcher, _results_erm)


def search_dj_bank(query_text,
                   INDEX_DIR,
                   fields=['title', 'body', 'path'],
                   limit_results=10,
                   b_bm25=.75,
                   k1_bm25=1.2):
    '''
    '''
    _ix = whoosh.index.open_dir(INDEX_DIR)
    _q = wh_search.parseQuery(_ix.schema, fields, query_text)
    _searcher = _ix.searcher(
        weighting=whoosh.scoring.BM25F(B=b_bm25, K1=k1_bm25)
    )  # The searching object and specifying that the custom scorer is used. (Default BM25f)
    _results_dj = wh_search.search_index(_searcher,
                                         _q,
                                         max_results=limit_results)

    return (_searcher, _results_dj)


def get_interface_results(query_text,
                          raw_data_path,
                          limit_results=50,
                          log_level=logging.DEBUG):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    _searcher, _results = search_over_index(query_text,
                                            dc.DATA_DIR,
                                            dc.INDEX_DIR,
                                            dc.MODEL_NAME,
                                            limit_results=limit_results)

    logger.info('Saving results to DataFrame.')
    _hit_ids = [_result['id'] for _result in _results]
    logger.debug('Matched Documents: {}'.format(_hit_ids))
    _raw_documents = pd.read_csv(raw_data_path)
    _raw_documents_hits = _raw_documents.iloc[_hit_ids]

    _interface_df = pd.DataFrame({
        'doc_id':
        _raw_documents_hits.index,
        'title':
        _raw_documents_hits['Title'],
        'path':
        _raw_documents_hits['Path'],
        'highlight_text': [None for _i in range(len(_raw_documents_hits))],
        'document':
        _raw_documents_hits['Document'],
        'last_updated':
        _raw_documents_hits['Last Updated']
    })
    logger.info('Finished Searching')
    return (_interface_df, _results)


if __name__ == "__main__":
    FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    query_text = 'banklån til privatkunde kunde'
    searcher_object, search_results = search_over_index(query_text,
                                                        dc.DATA_DIR,
                                                        dc.INDEX_DIR,
                                                        dc.MODEL_NAME,
                                                        a_sigmoid=10)
    logger.info([hit for hit in search_results])
