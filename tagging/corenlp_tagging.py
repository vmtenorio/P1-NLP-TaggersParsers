"""
Script to Perform Part of Speech tagging using the CoreNLP Java
wrapper from the Stanza library. I will be tagging using the
pre-trained models available in this library for the following languages:
    - English
    - Spanish

In english I will be tagging "Macbeth", from William Shakespeare,
available in the gutenberg corpus.

In spanish I will be tagging the novel "Niebla", from Miguel de Unamuno.
It is available in the Project Gutenberg repository:
http://gutenberg.org/ebooks/49836
http://gutenberg.org/files/49836/49836-0.txt

To run this script you must download the models for the languages specified
using the function stanza.install_corenlp() and then
stanza.download_corenlp_models() and the "gutenberg" package from the
NLTK library.

@author: Víctor Manuel Tenorio
"""
import nltk
import stanza
from stanza.server import CoreNLPClient
from utils import tups_to_file

# English
print("Starting tagging in English: Macbeth")
macbeth_raw = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
macbeth_words = nltk.word_tokenize(macbeth_raw)
macbeth_sents = nltk.sent_tokenize(macbeth_raw)

macbeth_corenlp_tags_tup = []
macbeth_corenlp_tags = []
with CoreNLPClient(
        annotators=['tokenize,pos'],
        timeout=30000,
        endpoint='http://localhost:9001',
        properties={'annotators': 'tokenize,pos'}) as client:
    # Doing in a loop to keep the same number of tokens
    for w in macbeth_words:
        corenlp_model = client.annotate(w)
        token = corenlp_model.sentence[0].token[0]
        macbeth_corenlp_tags_tup.append((token.word, token.pos))
        macbeth_corenlp_tags.append(token.pos)

tups_to_file('ingles/OUTPUT_CORENLP.txt', macbeth_corenlp_tags_tup)

print("Finished tagging in English: Macbeth")

# Spanish
# File obtained from http://gutenberg.org/ebooks/49836
# http://gutenberg.org/files/49836/49836-0.txt
print("Starting tagging in Spanish: Niebla")

with open('español/INPUT.txt', 'r') as f:
    niebla_raw = f.read()
    niebla_tokens = nltk.word_tokenize(niebla_raw)

niebla_corenlp_tags_tup = []
niebla_corenlp_tags = []

with CoreNLPClient(
        annotators=['openie'],
        timeout=30000,
        endpoint='http://localhost:9001',
        properties='spanish') as client:
    for w in niebla_tokens:
        corenlp_model = client.annotate(w)
        token = corenlp_model.sentence[0].token[0]
        niebla_corenlp_tags_tup.append((token.word, token.pos))
        niebla_corenlp_tags.append(token.pos)


tups_to_file('español/OUTPUT_CORENLP.txt', niebla_corenlp_tags_tup)
print("Finished tagging in Spanish: Niebla")

