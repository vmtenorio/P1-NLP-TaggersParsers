"""
Script to Perform Part of Speech tagging using Stanza library
I will be tagging using pre-trained models in four languages
natively supported by the pretrained models of the library:
    - English
    - Spanish
    - Finnish
    - Chinese

In english I will be tagging "Macbeth", from William Shakespeare,
available in the gutenberg corpus.

In spanish I will be tagging the novel "Niebla", from Miguel de Unamuno.
It is available in the Project Gutenberg repository:
http://gutenberg.org/ebooks/49836
http://gutenberg.org/files/49836/49836-0.txt

In finnish I will be tagging the novel Helsinkiin, by Juhani Aho,
available as well in the Project Gutenberg webpage:
http://gutenberg.org/ebooks/13580
http://gutenberg.org/cache/epub/13580/pg13580.txt

In chinese, I will be tagging the sentences of the corpus GSDSimp,
available in the GitHub repository from Universal Dependencies:
https://github.com/UniversalDependencies/UD_Chinese-GSDSimp

To run this script you must download the models for the languages specified
using the function stanza.download() and the "gutenberg" package from the
NLTK library for the macbeth text.

@author: Víctor Manuel Tenorio
"""

import nltk
import stanza
from sacremoses import MosesDetokenizer
from utils import tups_to_file, conll_text_reader

# English
print("Starting tagging in English: Macbeth")
macbeth_raw = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
macbeth_words = nltk.word_tokenize(macbeth_raw)
macbeth_sents = nltk.sent_tokenize(macbeth_raw)

config = {
    'processors': 'tokenize,pos', # Comma-separated list of processors to use     
    'lang': 'en', # Language code for the language to build the Pipeline in
    'use_gpu': False, # Configure it to run on GPU
    'tokenize_pretokenized': True # Use pretokenized text as input and disable tokenization
}

nlp_en = stanza.Pipeline(**config)
stanza_model = nlp_en([macbeth_words])

macbeth_stanza_tags_tup = []
macbeth_stanza_tags = []
for i in stanza_model.sentences:
    for j in i.words:
        macbeth_stanza_tags_tup.append((j.text, j.xpos))
        macbeth_stanza_tags.append(j.xpos)

tups_to_file('ingles/OUTPUT_STANZA.txt', macbeth_stanza_tags_tup)
print("Finished tagging in English: Macbeth")

# Spanish
# File obtained from http://gutenberg.org/ebooks/49836
# http://gutenberg.org/files/49836/49836-0.txt
print("Starting tagging in Spanish: Niebla")

with open('español/INPUT.txt', 'r') as f:
    niebla_raw = f.read()
    niebla_tokens = nltk.word_tokenize(niebla_raw)

config['lang'] = 'es'
nlp_es = stanza.Pipeline(**config)
stanza_model = nlp_es([niebla_tokens])

niebla_stanza_tags_tup = []
niebla_stanza_tags = []
for i in stanza_model.sentences:
    for j in i.words:
        niebla_stanza_tags_tup.append((j.text, j.xpos))
        niebla_stanza_tags.append(j.xpos)

tups_to_file('español/OUTPUT.txt', niebla_stanza_tags_tup)
tups_to_file('español/OUTPUT_STANZA.txt', niebla_stanza_tags_tup)

print("Finished tagging in Spanish: Niebla")

# File obtained from http://gutenberg.org/ebooks/13580
# http://gutenberg.org/cache/epub/13580/pg13580.txt
print("Starting tagging in Finnish: Helsinkiin")
with open('finlandes/INPUT.txt', 'r') as f:
    finnish_raw = f.read()

config['lang'] = 'fi'
nlp_fi = stanza.Pipeline(**config)
stanza_model = nlp_fi(finnish_raw)

finnish_stanza_tags_tup = []
finnish_stanza_tags = []
for i in stanza_model.sentences:
    for j in i.words:
        finnish_stanza_tags_tup.append((j.text, j.xpos))
        finnish_stanza_tags.append(j.xpos)

tups_to_file('finlandes/OUTPUT.txt', finnish_stanza_tags_tup)

print("Finished tagging in Finnish: Helsinkiin")

# Chinese
# Tagging from the corpus GSDSimp
print("Started tagging in Chinese: GSDSimp")
chinese_dep_parse = '../../dependency/UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-test.conllu'

with open(chinese_dep_parse, 'r') as gsdsimp_f:
    gsdsimp_text = conll_text_reader(gsdsimp_f)

detok = MosesDetokenizer()
with open('chino/INPUT.txt', 'w') as f:
    for s in gsdsimp_text:
        sent = detok.detokenize(s)
        f.write(sent + '\n')

nlp_zh = stanza.Pipeline(processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True, lang='zh')
stanza_model = nlp_zh(gsdsimp_text)

chinese_stanza_tags_tup = []
chinese_stanza_tags = []
for i in stanza_model.sentences:
    for j in i.words:
        chinese_stanza_tags_tup.append((j.text, j.xpos))
        chinese_stanza_tags.append(j.xpos)

tups_to_file('chino/OUTPUT.txt', chinese_stanza_tags_tup)

print("Finished tagging in Chinese: GSDSimp")
