from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import unicodedata
import string
import re
import random
from preprocessing import Lang

batch_size = 256
epochs = 100
latent_dim = 256
num_samples = 10000

def readData(reverse=False):
    """Read data, make pairs, normalize strings.
    --
    reverse -- whether to reverse source and target language
    """
    lang1 = "nl"
    lang2 = "en"
    print("Reading lines...")
    lines_input = open('./project/data/OpenSubtitles2016.en-nl.nl').read().strip().split('\n')
    lines_target =  open('./project/data/OpenSubtitles2016.en-nl.en').read().strip().split('\n')
    pairs = []
    for x, y in zip(lines_input[:num_samples], lines_target[:num_samples]):
        pairs.append([normalizeString(x), normalizeString(y)])
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def normalizeString(s):
    """Normalize a string. """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def unicodeToAscii(s):
    return ''.join(\
        c for c in unicodedata.normalize('NFD', s)\
            if unicodedata.category(c) != 'Mn'
        )

def prepareData(reverse=False):
    input_lang, output_lang, pairs = readData(reverse)
    print("Read %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

if __name__ == '__main__':
    prepareData()
