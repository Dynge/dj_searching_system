# Installation

```shell
> conda create -n ENV_NAME python=3.7 pip
> conda activate ENV_NAME
> pip install -r requirements.txt
```

# Initial setup

You will need to run some files to create the data files you will need.
The data you need is index' for the documents, a fasttext model and a normalization factor for the words in the vocabulary based on the similarity measure.

These can be create by running the following files:

```shell
> python whoosh_indexing
> python fasttext_training
> python precompute_p_vocabulary
```

# Workflow

At this point you can use the files for searching. The `searching_module` contains functions used for searching, both with query expansion and with a classic BM25.
The module `data_constants` contains constants used to access the data files. This way you do not have to hardcode paths as they are already saved as constants.

The code below shows a simple example of how to use these modules to complete a search.

```python
import searching_module as sm
import data_constants as dc

query = 'forretningsgang'
searcher_erm, results_erm = search_over_index(query_text=query, INDEX_DIR=dc.INDEX_DIR)

searcher_dj, results_dj = search_dj_bank(query_text=query, INDEX_DIR=dc.INDEX_DJ_DIR)
```
