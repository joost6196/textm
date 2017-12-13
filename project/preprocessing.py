from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

MAX_LENGTH = 20

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def unicodeToAscii(self, s):
        return ''.join(\
            c for c in unicodedata.normalize('NFD', s)\
                if unicodedata.category(c) != 'Mn'
            )

    def normalizeString(self, s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

class Preprocessing:
    def __init__(self, input_file, target_file, maxdata=10000):
        self.input_file = input_file
        self.target_file = target_file
        self.lang1 = "nl"
        self.lang2 = "en"
        self.maxdata = maxdata

    def readData(self, reverse=False):
        """Read data, make pairs, normalize strings.
        --
        reverse -- whether to reverse source and target language
        """
        print("Reading lines...")
        lines_input = open(self.input_file).read().strip().split('\n')
        lines_target =  open(self.target_file).read().strip().split('\n')
        lines_input = lines_input[:self.maxdata]
        lines_target = lines_target[:self.maxdata]
        pairs = []
        for x, y in zip(lines_input[:self.maxdata], lines_target[:self.maxdata]):
            pairs.append([self.normalizeString(x), self.normalizeString(y)])
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.lang2)
            output_lang = Lang(self.lang1)
        else:
            input_lang = Lang(self.lang1)
            output_lang = Lang(self.lang2)
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        pairs = self.filterPairs(pairs)
        return input_lang, output_lang, pairs

    def normalizeString(self, s):
        """Normalize a string. """
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def unicodeToAscii(self, s):
        return ''.join(\
            c for c in unicodedata.normalize('NFD', s)\
                if unicodedata.category(c) != 'Mn'
            )
    def filterPair(self, p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH


    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]
