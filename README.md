[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-green.svg)

# Smart-Chunker
This **smart chunker** is a semantic chunker to prepare a  long document for retrieval augmented generation (RAG).

Installing
----------


For installation, you need to Python 3.10 or later. You can install the **Smart-Chunker** from the [PyPi](https://pypi.org/project/smart-chunker) using the following command:

```
python -m pip install smart-chunker
```

If you want to install the **Smart-Chunker** in a Python virtual environment, you don't need to use `sudo`, but before installing, you will need to activate this virtual environment. For example, you can do this by using `conda activate your_favourite_environment` in the Linux terminal, or in the Anaconda Prompt for Microsoft Windows).

Also, 

To build this project from sources, you should run the following commands in the Terminal:

```
git clone https://github.com/bond005/smart_chunker.git
cd smart_chunker
python -m pip install .
```

In this case, you can run the unit tests to check workability and environment setting correctness:

```
python setup.py test
```

or

```
python -m unittest
```

For these tests, the `BAAI/bge-reranker-v2-m3` model will be used by default, and this model will be automatically downloaded from [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3). However, if your internet connection is unstable, you can download all the necessary files for this model and store them in the `tests/testdata/bge_reranker` subdirectory.

Usage
-----

After installing the **Smart-Chunker**, it can be used as a Python package in your projects. For example, you can create a new smart chunker for English using the [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and apply it to some English text as follows:

```python
from smart_chunker.chunker import SmartChunker

chunker = SmartChunker(
    language='en',
    reranker_name='BAAI/bge-reranker-v2-m3',
    newline_as_separator=False,
    device='cpu',
    max_chunk_length=20,
    minibatch_size=1,
    verbose=True
)
demo_text = 'There are many different approaches to Transformer fine-tuning. ' \
            'First, there is a development direction dedicated to the modification of ' \
            'the loss function and a specific problem statement. For example, training problem ' \
            'could be set as machine reading comprehence (question answering) instead of ' \
            'the standard sequence classification, or focal loss, dice loss and other things ' \
            'from other deep learning domains could be used instead of the standard ' \
            'cross-entropy loss function. Second, there are papers devoted to BERT extension, ' \
            'related to adding more input information from the knowledge graph, ' \
            'morpho-syntactic parsers and other things. Third, there is a group of algorithms ' \
            'associated with changing the learning procedure, such as metric learning ' \
            '(contrastive learning). Each direction has its own advantages and disadvantages, ' \
            'but the metric learning seems the most promising to us. Because the goal of ' \
            'any training is not to overfit the training sample and not just to take the top of ' \
            'the leaderboard on a particular test sample from the general population, ' \
            'but to ensure the highest generalization ability on the general population ' \
            'as a whole. High generalization ability is associated with good separation in ' \
            'the feature space. A good separation is possible when objects of ' \
            'different classes form sufficiently compact regions in our space. And methods of ' \
            'contrastive learning achieve better separation. Our goal is to test, on the basis of ' \
            'the RuNNE competition (Artemova et al., 2022), how true are these theoretical ' \
            'considerations in practice and how much will the use of comparative learning in BERT’s fine tuning allow us to build more compact high-level representations of different classes of named entities and, as a result, improve the quality of recognition of named entities.'
```

Demo
----

In the `demo` subdirectory you can see the **text_to_chunks.py** script and the **data** subdirectory.



License
-------

The **Smart-Chunker** (`smart-chunker`) is Apache 2.0 - licensed.