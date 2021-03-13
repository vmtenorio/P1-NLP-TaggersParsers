"""
Script to Perform Dependency Parsing using Stanza library
I will be tagging using pre-trained models in four languages
natively supported by the pretrained models of the library:
    - English
    - Spanish
    - Finnish
    - Chinese

In english I will be parsing the sentences from the builtin NLTK package
"dependency_treebank".

In spanish, I will be parsing the sentences from the AnCora corpus,
available in the GitHub repository from Universal Dependencies

In finnish, I will be parsing the sentences from the AnCora corpus,
available in the GitHub repository from Universal Dependencies

In chinese, I will be parsing the sentences from the AnCora corpus,
available in the GitHub repository from Universal Dependencies

To run this script you must download the models for the languages specified
using the function stanza.download() and the "dependency_treebank" package
from the NLTK library.

@author: Víctor Manuel Tenorio
"""

import nltk
import stanza
from stanza.utils.conll import CoNLL

from sacremoses import MosesDetokenizer
from conll18_ud_eval import evaluate, load_conllu
from sacremoses import MosesDetokenizer
from utils import print_results, conll_text_reader

import io


detok = MosesDetokenizer()

# English
gold_conll_en = ""
for s in nltk.corpus.dependency_treebank.parsed_sents()[:200]:
    gold_conll_en += s.to_conll(10) + '\r\n'

nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse')
stanza_conll_en = ""
for s in nltk.corpus.dependency_treebank.sents()[:200]:
    sent = detok.detokenize(s)
    doc = nlp(sent)
    for s in CoNLL.convert_dict(doc.to_dict()):
        for w in s:
            for i, content in enumerate(w):
                stanza_conll_en += content + '\t'
            stanza_conll_en = stanza_conll_en[:-1] + '\r\n'
        stanza_conll_en += '\r\n'

f_gold_en = io.StringIO(gold_conll_en)
f_stanza_en = io.StringIO(stanza_conll_en)

gold_en_eval = load_conllu(f_gold_en)
stanza_en_eval = load_conllu(f_stanza_en)

stanza_en_evaluation = evaluate(gold_en_eval, stanza_en_eval)

print_results(stanza_en_evaluation,
              "Results for Penn Treebank dataset using Stanza Dependency Parser")

# Spanish
spanish_dep_file = '../../dependency/UD_Spanish-AnCora-master/es_ancora-ud-test.conllu'

with open(spanish_dep_file, 'r') as ancora_f:
    ancora_text = conll_text_reader(ancora_f)

nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True, lang='es')
doc = nlp(ancora_text)
stanza_conll_es = ""
for s in CoNLL.convert_dict(doc.to_dict()):
    for w in s:
        for i, content in enumerate(w):
            stanza_conll_es += content + '\t'
        stanza_conll_es = stanza_conll_es[:-1] + '\r\n'
    stanza_conll_es += '\r\n'

with open(spanish_dep_file, 'r') as ancora_f:
    ancora_eval = load_conllu(ancora_f)

f_stanza_es = io.StringIO(stanza_conll_es)
stanza_es_eval = load_conllu(f_stanza_es)

stanza_es_evaluation = evaluate(ancora_eval, stanza_es_eval, turn_ascii=True)

print_results(stanza_es_evaluation,
              "Results for AnCora dataset using Stanza Dependency Parser")


# Finnish
finnish_dep_file = '../../dependency/UD_Finnish-TDT-master/fi_tdt-ud-test.conllu'

with open(finnish_dep_file, 'r') as tdt_f:
    tdt_text = conll_text_reader(tdt_f)

nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True, lang='fi')
doc = nlp(tdt_text)
stanza_conll_fi = ""
for s in CoNLL.convert_dict(doc.to_dict()):
    for w in s:
        for i, content in enumerate(w):
            stanza_conll_fi += content + '\t'
        stanza_conll_fi = stanza_conll_fi[:-1] + '\r\n'
    stanza_conll_fi += '\r\n'

with open(finnish_dep_file, 'r') as tdt_f:
    tdt_eval = load_conllu(tdt_f)

f_stanza_fi = io.StringIO(stanza_conll_fi)
stanza_fi_eval = load_conllu(f_stanza_fi)

stanza_fi_evaluation = evaluate(tdt_eval, stanza_fi_eval, check_charseq=False)

print_results(stanza_fi_evaluation,
              "Results for TDT dataset using Stanza Dependency Parser")

# Chinese
chinese_dep_parse = '../../dependency/UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-test.conllu'
with open(chinese_dep_parse, 'r') as gsdsimp_f:
    gsdsimp_text = conll_text_reader(gsdsimp_f)

nlp = stanza.Pipeline(processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True, lang='zh')
doc = nlp(gsdsimp_text)
stanza_conll_zh = ""
for s in CoNLL.convert_dict(doc.to_dict()):
    for w in s:
        for i, content in enumerate(w):
            stanza_conll_zh += content + '\t'
        stanza_conll_zh = stanza_conll_zh[:-1] + '\r\n'
    stanza_conll_zh += '\r\n'

with open(chinese_dep_parse, 'r') as gsdsimp_f:
    gsdsimp_eval = load_conllu(gsdsimp_f)

f_stanza_zh = io.StringIO(stanza_conll_zh)
stanza_zh_eval = load_conllu(f_stanza_zh)

stanza_zh_evaluation = evaluate(gsdsimp_eval, stanza_zh_eval)

print_results(stanza_zh_evaluation,
              "Results for GSDSimp dataset using Stanza Dependency Parser")


