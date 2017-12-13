from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import unicodedata
import string
import re
import random

batch_size = 128
epochs = 10
latent_dim = 256
num_samples = 10000
SOS_token = 0
EOS_token = 1

class Network:
    def __init__(self, input_lang, output_lang, pairs, savefilename):
        self.input_lang = input_lang
        self.savefilename = savefilename
        self.output_lang = output_lang
        self.input_texts = [] #with indexes of words
        self.target_texts = [] #with indexes of words, a sentence is a sequence
        for p in pairs:
            self.input_texts.append(self.variableFromSentence(input_lang, p[0]))
            self.target_texts.append(self.variableFromSentence(output_lang, p[1]))
        #input_words (these are in input_lang)
        #target_words (these are in output_lang)
        self.num_encoder_tokens = input_lang.n_words
        self.num_decoder_tokens = output_lang.n_words
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])
        self.encoder_input_data = np.zeros(\
            (len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),\
            dtype='float32')
        self.decoder_input_data = np.zeros(\
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),\
            dtype='float32')
        self.decoder_target_data = np.zeros(\
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),\
            dtype='float32')
        for i, (self.input_text, self.target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, word in enumerate(self.input_text):
                self.encoder_input_data[i, t, self.input_lang.word2index.get(word)] = 1.
            for t, char in enumerate(self.target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, self.output_lang.word2index.get(word)] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, self.output_lang.word2index.get(word)] = 1.

    def trainNetwork(self):
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        self.encoder = LSTM(latent_dim, return_state=True)
        self.encoder_outputs, self.state_h, self.state_c = self.encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [self.state_h, self.state_c]
        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                             initial_state=self.encoder_states)
        self.decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)
        # Save model
        model.save(self.savefilename)

    def inferenceNetwork(self):
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)
        self.decoder_state_input_h = Input(shape=(latent_dim,))
        self.decoder_state_input_c = Input(shape=(latent_dim,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_outputs, self.state_h, self.state_c = self.decoder_lstm(\
            self.decoder_inputs, initial_state=self.decoder_states_inputs)
        self.decoder_states = [self.state_h, self.state_c]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(\
            [self.decoder_inputs] + self.decoder_states_inputs,\
            [self.decoder_outputs] + self.decoder_states)

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.output_lang.word2index.get("SOS")] = 1.
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.output_lang.index2word.get(sampled_token_index)
            sampled_word = sampled_word + " "
            decoded_sentence += sampled_word
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_word == "EOS" or
               len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            # Update states
            states_value = [h, c]
        return decoded_sentence

    def makePredictions(self):
        for seq_index in range(20):
            # Take one sequence (part of the training test)
            # for trying out decoding.
            input_seq = self.encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq)
            print('-')
            inputsntc = [self.input_lang.index2word.get(word) for word in self.input_texts[seq_index]]
            inputstring = ''
            for x in inputsntc:
                if x != "EOS":
                    word = x + " "
                    inputstring += word
            print('Input sentence:', inputstring)
            print('Decoded sentence:', decoded_sentence)

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def variableFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return indexes

    def variablesFromPair(self, pair):
        input_variable = self.variableFromSentence(input_lang, pair[0])
        target_variable = self.variableFromSentence(output_lang, pair[1])
        return (input_variable, target_variable)
