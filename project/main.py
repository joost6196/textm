from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import unicodedata
import string
import re
import random
from preprocessing import Lang

batch_size = 32
epochs = 5
latent_dim = 256
num_samples = 10000
SOS_token = 0
EOS_token = 1

def readData(reverse=False):
    """Read data, make pairs, normalize strings.
    --
    reverse -- whether to reverse source and target language
    """
    lang1 = "nl"
    lang2 = "en"
    print("Reading lines...")
    lines_input = open('/home/pieter/projects/textm/project/data/OpenSubtitles2016.en-nl.nl').read().strip().split('\n')
    lines_target =  open('/home/pieter/projects/textm/project/data/OpenSubtitles2016.en-nl.en').read().strip().split('\n')
    lines_input = lines_input[:10000]
    lines_target = lines_target[:10000]
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
    print(random.choice(pairs))
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return indexes

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData()
    input_texts = [] #with indexes of words
    target_texts = [] #with indexes of words, a sentence is a sequence
    for p in pairs:
        input_texts.append(variableFromSentence(input_lang, p[0]))
        target_texts.append(variableFromSentence(output_lang, p[1]))
    #input_words (these are in input_lang)
    #target_words (these are in output_lang)
    num_encoder_tokens = input_lang.n_words
    num_decoder_tokens = output_lang.n_words
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    encoder_input_data = np.zeros(\
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),\
        dtype='float32')
    decoder_input_data = np.zeros(\
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),\
        dtype='float32')
    decoder_target_data = np.zeros(\
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),\
        dtype='float32')
    print("Generating encoder input and decoder input.")
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, word in enumerate(input_text):
            encoder_input_data[i, t, input_lang.word2index.get(word)] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, output_lang.word2index.get(word)] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, output_lang.word2index.get(word)] = 1.

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
    # Save model
    model.save('s2swordbasedmodel.h5')
