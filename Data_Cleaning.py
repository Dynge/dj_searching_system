# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:02:43 2020
@author: Mathias Huus Olsen
"""

### Libraries ###

import os
import re
import pandas as pd
import logging
import data_constants as dc

FORMAT = '%(asctime)s - ' '%(levelname)s - ' '%(filename)s:' '%(funcName)s() - ' '%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger('data_cleaning.py')

### Functions ###


def import_text_file(_text_name, _text_directory):
    '''
    Function for importing text files

    Input:
        _text_name is the name of the text document as a string
        _text_directory is the document directory as a string

    Output:
        _text_file is the imported text document as a string
    '''
    os.chdir(_text_directory)
    _text_file = open(_text_name, 'r', encoding='cp1254').read()
    logger.debug('The first 100 characters of the file: "{}"...'.format(
        _text_file[:100]))
    return (_text_file)


def find_item(_pattern, _string, _type, _dot_all=0):
    '''
    Function for finding information and its position in a string with patterns
        Input
            _pattern is the pattern which you want to find in the string as a string
            _string is the string inwhich you want to find patterns
            _type is the type of information you are looking for as a string
            _dot_all is 0 as standard, it can be changed to re.DOTALL for finding document content. Looks in multiple sentences
        Output
            _item_data is a dataframe containing the information found with the patterns, its positions and the type provided input
                Columns: Position, Type, Value
    '''

    # Finding position of patterns in text document
    _item_position = []
    _match = re.finditer(_pattern, _string, _dot_all)
    for _m in _match:
        _item_position.append(_m.start())

    # Finding information based on pattern
    _item_value = re.findall(_pattern, _string, _dot_all)

    # Used to get numbers before sections and chapters
    if _type == "Section":
        _section_numb = re.findall(r'\nSectionNumb:\s\s(.*?)\n', _string)
        _item_value_new = [(_section_numb[i] + ". " + _item_value[i])
                           for i in list(range(0, len(_item_value)))]
    elif _type == "Chapter":
        _section_position = [((_string[0:i]).rfind('\nSectionNumb:  '))
                             for i in _item_position]
        _slen = len('\nSectionNumb:  ')
        _section_numb = [(_string[(i + _slen):(i + _slen + 2)].strip('\n'))
                         for i in _section_position]
        _chapter_numb = re.findall(r'\nChapterNumb:\s\s(.*?)\n', _string)
        _item_value_new = [
            (_section_numb[i] + "." + _chapter_numb[i] + " " + _item_value[i])
            for i in list(range(0, len(_item_value)))
        ]
    elif _type == "SubChapter":
        _section_position = [((_string[0:i]).rfind('\nSectionNumb:  '))
                             for i in _item_position]
        _slen = len('\nSectionNumb:  ')
        _section_numb = [(_string[(i + _slen):(i + _slen + 2)].strip('\n'))
                         for i in _section_position]
        _chapter_position = [((_string[0:i]).rfind('\nChapterNumb:  '))
                             for i in _item_position]
        _clen = len('\nChapterNumb:  ')
        _chapter_numb = [(_string[(i + _clen):(i + _clen + 2)].strip('\n'))
                         for i in _chapter_position]
        _subchapter_numb = re.findall(r'\nSubChapterNumb:\s\s(.*?)\n', _string,
                                      _dot_all)
        _item_value_new = [(_section_numb[i] + "." + _chapter_numb[i] + "." +
                            _subchapter_numb[i] + " " + _item_value[i])
                           for i in list(range(0, len(_item_value)))]
    elif _type == "MinorChapter":
        _section_position = [((_string[0:i]).rfind('\nSectionNumb:  '))
                             for i in _item_position]
        _slen = len('\nSectionNumb:  ')
        _section_numb = [(_string[(i + _slen):(i + _slen + 2)].strip('\n'))
                         for i in _section_position]
        _chapter_position = [((_string[0:i]).rfind('\nChapterNumb:  '))
                             for i in _item_position]
        _clen = len('\nChapterNumb:  ')
        _chapter_numb = [(_string[(i + _clen):(i + _clen + 2)].strip('\n'))
                         for i in _chapter_position]
        _subchapter_position = [((_string[0:i]).rfind('\nSubChapterNumb:  '))
                                for i in _item_position]
        _sclen = len('\nSubChapterNumb:  ')
        _subchapter_numb = [
            (_string[(i + _sclen):(i + _sclen + 2)].strip('\n'))
            for i in _subchapter_position
        ]
        _minorchapter_numb = re.findall(r'\nMinorChapterNumb:\s\s(.*?)\n',
                                        _string, _dot_all)
        _item_value_new = [(_section_numb[i] + "." + _chapter_numb[i] + "." +
                            _subchapter_numb[i] + "." + _minorchapter_numb[i] +
                            " " + _item_value[i])
                           for i in list(range(0, len(_item_value)))]
    else:
        _item_value_new = _item_value

    _item_type = [_type] * len(_item_position)

    _item_data = pd.DataFrame(list(
        zip(_item_position, _item_type, _item_value_new)),
                              columns=['Position', 'Type', 'Value'])

    return (_item_data)


def clean_data(_dataframe):
    '''
    Function for sorting and cleaning dataframes made from the previously found patterns. Empty documents are removed as well.
    Input
        _dataframe is a dataframe with information about position, type and document content
            Columns: Position, Type, Document
    Output
        _cleaned_data is a dataframe without empty documents and divided into the categories and sub categories
            Columns: System, Category, Section, Chapter, SubChapter, MinorChapter, Document, Time
    '''
    _cleaned_list = []
    logger.info('Cleaning the data of a dataframe...')
    # Dividing the information into different variables which can be used as columns in dataframe
    for i in list(range(0, len(_dataframe.iloc[:, 0]))):
        if _dataframe.iloc[i, 1] == "Time":
            _time = _dataframe.iloc[i, 2]
            _category = ""
            _section = ""
            _chapter = ""
            _subchapter = ""
            _minorchapter = ""
            _document = ""
        elif _dataframe.iloc[i, 1] == "Category":
            _category = _dataframe.iloc[i, 2]
        elif _dataframe.iloc[i, 1] == "Section":
            _section = _dataframe.iloc[i, 2]
        elif _dataframe.iloc[i, 1] == "Chapter":
            _chapter = _dataframe.iloc[i, 2]
        elif _dataframe.iloc[i, 1] == "SubChapter":
            _subchapter = _dataframe.iloc[i, 2]
        elif _dataframe.iloc[i, 1] == "MinorChapter":
            _minorchapter = _dataframe.iloc[i, 2]
        elif _dataframe.iloc[i, 1] == "Document":
            _document = _dataframe.iloc[i, 2]
            _cleaned_list.append([
                _dataframe.iloc[i, 3], _category, _section, _chapter,
                _subchapter, _minorchapter, _document, _time
            ])

    _cleaned_data = pd.DataFrame(_cleaned_list)
    _cleaned_data.columns = [
        "System", "Category", "Section", "Chapter", "SubChapter",
        "MinorChapter", "Document", "Time"
    ]

    return (_cleaned_data)


def finish_data(_cleaned_data):
    '''
    Function for presenting the cleaned dataset´as it will be used in the searching system. Titles and paths are found of the documents
    Input
        _cleaned_data is the cleaned dataset from the clean_data() function
            #Columns: System, Category, Section, Chapter, SubChapter, MinorChapter, Document, Time
    Output
        _finished_data contains the paths and titles of the documents as well as document content and when it was last updated
            Columns: Title, Path, Document, Last Updated
    '''
    _finished_list = []

    # Saving title and path variables depending on the categories and subcategories for each document
    for i in list(range(0, len(_cleaned_data.iloc[:, 0]))):

        if _cleaned_data.iloc[i, 5] != "":
            _title = _cleaned_data.iloc[i, 5]
            _path = (_cleaned_data.iloc[i, 0] + " > " +
                     _cleaned_data.iloc[i, 1] + " > " +
                     _cleaned_data.iloc[i, 2] + " > " +
                     _cleaned_data.iloc[i, 3] + " > " +
                     _cleaned_data.iloc[i, 4])
        elif _cleaned_data.iloc[i, 4] != "":
            _title = _cleaned_data.iloc[i, 4]
            _path = (_cleaned_data.iloc[i, 0] + " > " +
                     _cleaned_data.iloc[i, 1] + " > " +
                     _cleaned_data.iloc[i, 2] + " > " +
                     _cleaned_data.iloc[i, 3])
        elif _cleaned_data.iloc[i, 3] != "":
            _title = _cleaned_data.iloc[i, 3]
            _path = (_cleaned_data.iloc[i, 0] + " > " +
                     _cleaned_data.iloc[i, 1] + " > " +
                     _cleaned_data.iloc[i, 2])
        elif _cleaned_data.iloc[i, 2] != "":
            _title = _cleaned_data.iloc[i, 2]
            _path = (_cleaned_data.iloc[i, 0] + " > " +
                     _cleaned_data.iloc[i, 1])

        _finished_list.append([
            _title, _path, _cleaned_data.iloc[i, 6], _cleaned_data.iloc[i, 7]
        ])

    _finished_data = pd.DataFrame(_finished_list)
    _finished_data.columns = ["Title", "Path", "Document", "Last Updated"]

    return (_finished_data)


#### The program script ####

### Import Text Files ###

logger.info('Importing data from {}'.format(dc.DATA_DIR))
business_raw = import_text_file(dc.BUSINESS_NAME, dc.DATA_DIR)

product_raw = import_text_file(dc.PRODUCT_NAME, dc.DATA_DIR)

### Data Cleaning ###
## Finding information from patterns in the business procedures ##
logger.info('Identifying patterns in the business procedures...')
business_docs = find_item(r'Body:\s\s(.*?)\x0c', business_raw, "Document",
                          re.DOTALL)

business_times = find_item(r'Modified:\s\s(.*?)\n', business_raw, "Time")

business_categories = find_item(r'\nCategories:\s\s(.*?)\n', business_raw,
                                "Category")

business_sections = find_item(r'\nSectionTitle:\s\s(.*?)\n', business_raw,
                              "Section")

business_chapter = find_item(r'\nChapterTitle:\s\s(.*?)\n', business_raw,
                             "Chapter")

business_subchapter = find_item(r'\nSubChapterTitle:\s\s(.*?)\n', business_raw,
                                "SubChapter")

business_minorchapter = find_item(r'\nMinorChapterTitle:\s\s(.*?)\n',
                                  business_raw, "MinorChapter")

# Combine information into a single dataset and sorting after position #
business_procedures = pd.concat([
    business_docs, business_times, business_categories, business_sections,
    business_chapter, business_subchapter, business_minorchapter
])

sorted_bp = business_procedures.sort_values(by=["Position"])

sorted_bp["System"] = "Forretningsgang"

## Finding information from patterns in the products ##
logger.info('Identifying patterns in the products...')
product_docs = find_item(r'Body:\s\s(.*?)\x0c', product_raw, "Document",
                         re.DOTALL)

product_times = find_item(r'Modified:\s\s(.*?)\n', product_raw, "Time")

product_categories = find_item(r'\nCategories:\s\s(.*?)\n', product_raw,
                               "Category")

product_sections = find_item(r'\nSectionTitle:\s\s(.*?)\n', product_raw,
                             "Section")

product_chapters = find_item(r'\nChapterTitle:\s\s(.*?)\n', product_raw,
                             "Chapter")

product_subchapters = find_item(r'\nSubChapterTitle:\s\s(.*?)\n', product_raw,
                                "SubChapter")

# Combine information into a single dataset and sorting after position #
products = pd.concat([
    product_docs, product_times, product_categories, product_sections,
    product_chapters, product_subchapters
])

sorted_products = products.sort_values(by=["Position"])

sorted_products["System"] = "Produktoversigt"


## Removing empty documents and create columns for categoríes and subcategories ##
cleaned_bp = clean_data(sorted_bp)
cleaned_products = clean_data(sorted_products)

# Combining business procedure and product dataframes #
cleaned_data = pd.concat([cleaned_bp, cleaned_products])

# Presenting dataframe with title and path instead of categories and subcategories
finished_data = finish_data(cleaned_data)

# Exporting dataframe as excel file
logger.info('Saving the data in "{}"'.format(dc.DATA_DIR))
if not os.path.exists(dc.DATA_DIR):
    os.mkdir(dc.DATA_DIR)
#finished_data.to_excel(dc.DATA_DIR+"/"+OUTPUT_NAME,header = True)

finished_data.to_csv(dc.DATA_DIR + dc.RAW_DATA_NAME, header=True)
