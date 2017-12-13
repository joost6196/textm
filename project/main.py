from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import unicodedata
import string
import re
import random
from preprocessing import Lang, Preprocessing
from neuralkeras import Network

def printInfo(input_lang, output_lang, pairs):
    print("Read %s sentence pairs" % len(pairs))
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(random.choice(pairs))

if __name__ == '__main__':
    input_file = '/home/pieter/projects/textm/project/data/OpenSubtitles2016.en-nl.nl'
    target_file = '/home/pieter/projects/textm/project/data/OpenSubtitles2016.en-nl.en'
    preproc = Preprocessing(input_file, target_file)
    input_lang, output_lang, pairs = preproc.readData()
    printInfo(input_lang, output_lang, pairs)
    network = Network(input_lang, output_lang, pairs, "model10k.h5")
    network.trainNetwork()
    network.inferenceNetwork()
    network.makePredictions()
