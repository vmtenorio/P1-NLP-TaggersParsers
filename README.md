# Tagging and Parsing using Python

This is the repository for the first assignment of the subjet Empirical Methods of Natural Language Processing of the Master in Artificial Intelligence Research from the UIMP - AEPIA.

It contains all the necessary code to perform PoS tagging and constituency and dependency parsing of the text of some standard corpora available within the NLTK Python library. For this task, it uses both the NLTK and Stanza libraries. Moreover, it uses CoreNLP Java client using the wrapper class `CoreNLPClient` available in the Stanza library.

The repository is structured as follows:
* The `tagging` directory contains the necessary code to PoS tag text in different languages. It either uses corpora available within NLTK packages or the INPUT files in the directories for each language.
* The `cons_parsing` directory contains the code to perform constituency parsing of text in spanish and english.
* The `dep_parsing` directory contains the necessary code to perform the dependency parsing of the text. It also contains an slightly modified version of the evaluation script for the CoNLL Shared task (see [link](http://universaldependencies.org/conll18/evaluation.html)).
* The `NLP_P1.ipynb` is a Jupyter notebook with all the code of the tasks combined in a unique notebook.

