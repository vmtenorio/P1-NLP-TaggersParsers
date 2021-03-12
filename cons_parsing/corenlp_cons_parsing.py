"""
Script to Perform Constuency parsing using the CoreNLP wrapper
from the Stanza library I will be parsing sentences in the
following languages:
    - English
    - Spanish

In english I will be tagging the parsed sentences from the Penn 
treebank, available natively in the library NLTK, in the treebank
package

In english I will be tagging the parsed sentences from the Cess-ESP
treebank, also available natively in the library NLTK, in the cess_esp
package

To run this script you must download the models for the languages specified
using the function stanza.install_corenlp(), the models for the spanish
language from this same library and the "treebank" and "cess_esp"
packages from the NLTK library for the macbeth text.

The evaluation of the parsing will be performed using EVALB tool,
available in the PYEVALB python package

@author: Víctor Manuel Tenorio
"""

import nltk
import stanza
from stanza.server import CoreNLPClient
from sacremoses import MosesDetokenizer
from PYEVALB import scorer, parser

import numpy as np
import re

detok = MosesDetokenizer()
evalb_scorer = scorer.Scorer()

recalls_corenlp = []
precs_corenlp = []
accs_corenlp = []
parsed_sents = nltk.corpus.treebank.parsed_sents()
skipped_sents = 0
sents_analyzed = 0
with CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'parse'],
        output_format="json",
        timeout=3000001,
        endpoint='http://localhost:9001') as client:
    for i, s in enumerate(nltk.corpus.treebank.sents()):
        sent = detok.detokenize(s)
        corenlp_model = client.annotate(sent)
        gold_sent = parser.create_from_bracket_string(parsed_sents[i].pformat())
        parse_tree = parser.create_from_bracket_string(corenlp_model['sentences'][0]['parse'])
        try:
            scores = evalb_scorer.score_trees(gold_sent, parse_tree)
        except:
            skipped_sents += 1
            continue
        recalls_corenlp.append(scores.recall)
        precs_corenlp.append(scores.prec)
        accs_corenlp.append(scores.tag_accracy)
        sents_analyzed += 1
        if sents_analyzed == 100:
            break

print("Results of the constituency parsing by CoreNLP in english")
print("Accuracy: " + str(np.mean(accs_corenlp)))
print("Precision: " + str(np.mean(recalls_corenlp)))
print("Recall: " + str(np.mean(precs_corenlp)))

# Spanish
recalls_corenlp = []
precs_corenlp = []
accs_corenlp = []
parsed_sents = nltk.corpus.cess_esp.parsed_sents()
skipped_sents = 0
sents_analyzed = 0
with CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'parse'],
        output_format="json",
        timeout=3000001,
        endpoint='http://localhost:9001',
        properties='spanish') as client:
    for i, s in enumerate(nltk.corpus.cess_esp.sents()):
        sent = detok.detokenize(s)
        corenlp_model = client.annotate(sent)
        # Adequating parsing sent
        parsed_s = re.sub('grup\.nom\.[a-z]*', 'grup.nom', parsed_sents[i].pformat())
        parsed_s = re.sub('s\.a\.[a-z]*', 's.a', parsed_s)
        parsed_s = re.sub('grup\.a\.[a-z]*', 'grup.a', parsed_s)
        parsed_s = re.sub('espec\.[a-z]*', 'espec', parsed_s)
        parsed_s = re.sub('conj\.[a-z]*', 'conj', parsed_s)
        parsed_s = re.sub('{\(Fe|\(Fc|\(Fp}', '(PUNCT', parsed_s)
        gold_sent = parser.create_from_bracket_string(parsed_s)
        parse_tree = parser.create_from_bracket_string(corenlp_model['sentences'][0]['parse'])
        try:
            scores = evalb_scorer.score_trees(gold_sent, parse_tree)
        except:
            skipped_sents += 1
            continue
        recalls_corenlp.append(scores.recall)
        precs_corenlp.append(scores.prec)
        accs_corenlp.append(scores.tag_accracy)
        sents_analyzed += 1
        if skipped_sents == 1000:
            print("Skipped with " + str(sents_analyzed))
            break
        if sents_analyzed == 100:
            break

print("Results of the constituency parsing by CoreNLP in spanish")
print("Accuracy: " + str(np.mean(accs_corenlp)))
print("Precision: " + str(np.mean(recalls_corenlp)))
print("Recall: " + str(np.mean(precs_corenlp)))

