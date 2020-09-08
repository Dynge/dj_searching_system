import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data/'
BUSINESS_NAME = "Forretningsgang_Struktured_Text.txt"
PRODUCT_NAME = "Produkter_Struktured_Text.txt"
RAW_DATA_NAME = 'dj_bank_documents.csv'
LEMMA_NAME = 'dj_bank_lemmatized.csv'
STOP_NAME = 'dj_bank_stopandlemma.csv'

MODEL_NAME = 'fasttext_model.vec'

DATA_NAME = 'dj_bank_processed.csv'
MLE_WORDS = 'mle_words.npy'
INDEX_DIR = os.path.dirname(os.path.abspath(__file__)) + '/indexdir/'
INDEX_DJ_DIR = os.path.dirname(os.path.abspath(__file__)) + '/indexdir_dj/'

IMAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/images/'
IMAGE_ARROW_RIGHT = "arrow_right.png"
IMAGE_ARROW_LEFT = "arrow_left.png"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/output/'
QUERY_MICHAEL_NAME = "query_data_Michael.csv"
QUERY_MATHIAS_NAME = "query_data_Mathias.csv"
QUERY_VALIDATION_NAME = 'query_validation.csv'
QUERY_TEST_NAME = 'query_test.csv'
QUERY_VAL_SEARCH_RESULTS_NAME = 'query_val_search_results.csv'
QUERY_TEST_SEARCH_RESULTS_NAME = 'query_test_search_results.csv'
QUERY_DOCUMENT_RELEVANCE_MATHIAS = 'relevancy_data_Mathias.csv'
QUERY_DOCUMENT_RELEVANCE_MICHAEL = 'relevancy_data_Michael.csv'
INTERFACE_DATA = "interface_data.csv"

QREL_NAME = 'qrel_data.csv'

SCREENSHOT_RANKS = 'screenshot_ranks_dj_search.csv'
COMPARISON_CSV = 'comparison_dj_bank_search_results.csv'

RAW_PROCEDURES_NAMES = 'dj_bank_procedures.csv'
INDEX_PRODUCTS_LOWERED_DIR = os.path.dirname(
    os.path.abspath(__file__)) + '/indexdir_raw_lower/'
INDEX_PRODUCTS_CLEANED_DIR = os.path.dirname(
    os.path.abspath(__file__)) + '/indexdir_raw_clean/'

TUNE_HYP_NAME = 'hyperparameter_tuning_results.csv'
TUNE_HYP_SEARCHES = 'hyperparameter_tuning_searches.npy'


def get_p_vocab_name(a_sigmoid):
    '''
    This constant is dependant on the a parameter of the sigmoid function.
    Therefore in order to get the PATH, you must input the sigmoid parameter you're using.
    '''
    _p_vocab_name = 'p_word_vocabulary_asigmoid_'
    _extension = '.npy'
    return (_p_vocab_name + str(a_sigmoid) + _extension)
