import logging
import pickle

def save_data_to_file( data, fname ):
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    with open( fname, "wb" ) as fp:   #Pickling
        pickle.dump( data, fp )
    logger.info( 'Saved data to {}.'.format( fname ) )
    return( True )

def read_data_from_file( fname ):   
    '''
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    with open( fname, "rb" ) as fp:   # Unpickling
        _read_data = pickle.load(fp)
    logger.info( 'Loaded data from {}.'.format( fname ) )
    return( _read_data )
