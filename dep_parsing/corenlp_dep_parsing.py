"""
Script to Perform Dependency Parsing using the CoreNLP wrapper
from the Stanza library I will be tagging using pre-trained models
in two languages natively supported by the pretrained models of the library:
    - English
    - Spanish

In english I will be parsing the sentences from the builtin NLTK package
"dependency_treebank".

In spanish, I will be parsing the sentences from the AnCora corpus,
available in the GitHub repository from Universal Dependencies

To run this script you must download the models for the languages specified
using the function stanza.download() and the "dependency_treebank" package
from the NLTK library.

To run this script you must download the models for the languages specified
using the function stanza.install_corenlp(), the models for the spanish
language from this same library and the "treebank" and "cess_esp"
packages from the NLTK library for the macbeth text.

The evaluation of the performance of the algorithm will be performed using the
evaluation script from the CoNLL 2018 Shared task, available at:
http://universaldependencies.org/conll18/evaluation.html

@author: Víctor Manuel Tenorio
"""

import nltk
from stanza.server import CoreNLPClient

from sacremoses import MosesDetokenizer
from conll18_ud_eval import evaluate, load_conllu
from utils import print_results, conll_text_reader

import io

detok = MosesDetokenizer()

# English
gold_conll_en = ""
for s in nltk.corpus.dependency_treebank.parsed_sents()[:200]:
    gold_conll_en += s.to_conll(10) + '\r\n'

corenlp_conll_en = ""
with CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'parse', 'depparse'],
        output_format="conllu",
        timeout=3000001,
        endpoint='http://localhost:9001') as client:
    for s in nltk.corpus.dependency_treebank.sents()[:200]:
        sent = detok.detokenize(s)
        corenlp_model = client.annotate(sent)
        corenlp_conll_en += corenlp_model# + '\r\n'

f_corenlp_en = io.StringIO(corenlp_conll_en.replace("Corp.", "Corp").replace("Conn.", "Conn").replace("Â", "").replace("Ltd.", "Ltd"))
corenlp_en_eval = load_conllu(f_corenlp_en)

f_gold_en = io.StringIO(gold_conll_en.replace("Corp.", "Corp").replace("Conn.", "Conn").replace("Ltd.", "Ltd"))
gold_en_eval = load_conllu(f_gold_en)
corenlp_en_evaluation = evaluate(gold_en_eval, corenlp_en_eval)

print_results(corenlp_en_evaluation,
              "Results for Dependency Treebank dataset using CoreNLP Dependency Parser")

# Spanish
spanish_dep_file = '../../dependency/UD_Spanish-AnCora-master/es_ancora-ud-test.conllu'

with open(spanish_dep_file, 'r') as ancora_f:
    ancora_text = conll_text_reader(ancora_f)

corenlp_conll_es = ""
with CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'parse', 'depparse'],
        output_format="conllu",
        timeout=3000001,
        endpoint='http://localhost:9001',
        properties='spanish') as client:
    for s in ancora_text:
        sent = detok.detokenize(s)
        corenlp_model = client.annotate(sent)
        corenlp_conll_es += corenlp_model# + '\r\n'

with open(spanish_dep_file, 'r') as ancora_f:
    ancora_eval = load_conllu(ancora_f)

f_corenlp_es = io.StringIO(corenlp_conll_es)
corenlp_es_eval = load_conllu(f_corenlp_es)
corenlp_es_evaluation = evaluate(ancora_eval, corenlp_es_eval, check_charseq=False)

print_results(corenlp_es_evaluation,
              "Results for AnCora dataset using CoreNLP Dependency Parser")

