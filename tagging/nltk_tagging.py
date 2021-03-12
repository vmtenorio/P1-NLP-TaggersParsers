"""
Script to Perform Part of Speech tagging using NLTK library
First of all I will be tagging using pre-trained models in the two
languages natively supported by the pretrained function pos_tag:
    - English
    - Russian

Then I will use train my own TrigramTaggers and a simple NN tagger, using
Word2Vec representations of the words. To train these taggers, I will
use the tagged words from the Penn treebank, available natively in the
treebank package from NLTK.

In english I will be tagging "Macbeth", from William Shakespeare,
available in the gutenberg corpus.

In russian I will be tagging the novel Moscovia v predstavlenii
inostrantsev XVI-XVII, written by Pavel Nikolayevich Apostol. It is
available in the Project Gutenberg webpage, at:
http://gutenberg.org/ebooks/30774
http://gutenberg.org/files/30774/30774-0.txt

@author: Víctor Manuel Tenorio

Dependency NLTK packages for this script:
    - gutenberg
    - averaged_perceptron_tagger
    - averaged_perceptron_tagger_ru
    - treebank
"""
import nltk
from utils import tups_to_file

from gensim.models import Word2Vec
import numpy as np
import keras
from sklearn.model_selection import train_test_split

############################################################
#################### Pre-Trained Models ####################
############################################################

# English tagging Macbeth
macbeth_raw = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
macbeth_words = nltk.word_tokenize(macbeth_raw)
macbeth_sents = nltk.sent_tokenize(macbeth_raw)

with open('ingles/INPUT.txt', 'w') as f:
    f.write(macbeth_raw)

macbeth_tags_nltk_tup = nltk.pos_tag(macbeth_words)
macbeth_tags_nltk = [x[1] for x in macbeth_tags_nltk_tup]

tups_to_file('ingles/OUTPUT_NLTK.txt', macbeth_tags_nltk_tup)

# Russian
# File obtained from http://gutenberg.org/ebooks/30774
# http://gutenberg.org/files/30774/30774-0.txt
with open('ruso/INPUT.txt', 'r') as f:
    russian_tokens = nltk.word_tokenize(f.read())

russian_tags = nltk.pos_tag(russian_tokens, lang='rus')

tups_to_file('ruso/OUTPUT.txt', russian_tags)


############################################################
################### Training the Models ####################
############################################################

# Gathering the training data from Penn Treebank
penn_sents = nltk.corpus.treebank.tagged_sents()
penn_sents_train, penn_sents_test = train_test_split(penn_sents, test_size=0.15)

# TrigramTagger
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(penn_sents_train, backoff=t0)
t2 = nltk.BigramTagger(penn_sents_train, backoff=t1)
t3 = nltk.TrigramTagger(penn_sents_train, backoff=t2)

print("Accuracy of the trigramm tagger in a Penn Treebank random test set: ", end="")
print(t3.evaluate(penn_sents_test))

macbeth_trigram_predictions = t3.tag(macbeth_words)
macbeth_trigram_tags = [x[1] for x in macbeth_trigram_predictions]

assert len(macbeth_tags_nltk) == len(macbeth_trigram_tags)
coinc = [macbeth_tags_nltk[i] == macbeth_trigram_tags[i] for i in range(len(macbeth_trigram_tags))]
print("Percentage of tags that are coincident with NLTK default predictor: ", end="")
print(100*sum(coinc) / len(macbeth_tags_nltk))

# Saving to file
tups_to_file('ingles/OUTPUT_TrigramTagger.txt', macbeth_trigram_predictions)

# Neural network tagger
penn_sents = nltk.corpus.treebank.sents()
penn_tagged_words = nltk.corpus.treebank.tagged_words()

penn_tags = list(set([x[1] for x in penn_tagged_words]))

model = Word2Vec(penn_sents, min_count=1, size=100, window=5)

nn_model = keras.Sequential(
    [
        keras.layers.Dense(100, activation="relu", name="layer1"),
        keras.layers.Dense(100, activation="relu", name="layer2"),
        keras.layers.Dense(len(penn_tags), activation="softmax", name="output"),
    ]
)

nn_model.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['accuracy'])

x = []
y = []
for w in penn_tagged_words:
    if w[0] in model.wv:
        x.append(model.wv[w[0]])
        y.append(penn_tags.index(w[1]))
x = np.array(x)
y = np.array(y)
y = keras.utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

nn_model.fit(x_train, y_train, epochs=50, batch_size=32)

# Evaluating on the Penn treebank random test set
penn_eval = nn_model.evaluate(x_test, y_test)
print("Penn Treebank evaluation")
print("Loss: {} - Accuracy: {}".format(penn_eval[0], 100*penn_eval[1]))

# Predicting for Macbeth
macbeth_model = Word2Vec(macbeth_sents, min_count=1,
                         size=100, window=5)

x_macbeth = []
for w in macbeth_words:
    if w[0] in macbeth_model.wv:
        x_macbeth.append(macbeth_model.wv[w[0]])
x_macbeth = np.array(x_macbeth)

yhat_macbeth = nn_model.predict(x_macbeth)

macbeth_nn_tags_idx = np.argmax(yhat_macbeth, axis=1)
macbeth_nn_tags = [penn_tags[i] for i in macbeth_nn_tags_idx]
macbeth_nn_tags_tup = [(macbeth_words[i], penn_tags[j]) for i, j in enumerate(macbeth_nn_tags_idx)]

assert len(macbeth_tags_nltk) == len(macbeth_nn_tags)
coinc = [macbeth_tags_nltk[i] == macbeth_nn_tags[i] for i in range(len(macbeth_nn_tags))]
print("Percentage of tags that are coincident with NLTK default predictor: ", end="")
print(100*sum(coinc) / len(macbeth_tags_nltk))

tups_to_file('ingles/OUTPUT_NNTagger.txt', macbeth_nn_tags_tup)
