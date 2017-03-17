# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:58:18 2017

@author: gopal
"""

import json
import os
import numpy as np
import glob
import nltk.data
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.layers.recurrent import GRU
from keras.layers.core import RepeatVector, Masking, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from nltk import FreqDist

# In[]
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
#from nltk.corpus import brown, movie_reviews, treebank
#b = Word2Vec(brown.sents())
#b['learn']

STORY_BASE_FOLDER = './books_txt_full/Romance/'
OUTPUT_FILE = 'corpus.txt'
OUTPUT_NP_FILE = 'corpus'
OUTPUT_NP_W2V = 'corpus_w2v'
MAX_SEQ_LEN = 30
PAD_VAL = 0.0
HIDDEN_UNITS = 400

def create_corpus(limit = 0):
    corpus = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    i=0
    for file in (glob.glob(STORY_BASE_FOLDER + '*.txt')):
        print ("Reading " + str(i))
        i+=1
        text = open(file, errors='ignore').read()
        sents = tokenizer.tokenize(text)
        
        for sent in sents[100:150]:
            sent_words = word_tokenize(sent)
            if len(sent_words) >= 2 and  len(sent_words) <= 70:
                corpus.append(sent_words)
        if i>limit and limit>0:
            break
    return corpus
        
def corpus_to_file(corpus):
    with open(OUTPUT_FILE, 'w') as f:
        for sent in corpus:
            words =  ' '.join(sent)
            f.write(words+ '\n')
            
def corpusfile_to_vec():
    corpus=[]
    sents = open(OUTPUT_FILE, errors='ignore').readlines()
    i=0
    for l in sents:
        print ("Reading " + str(i))
        i+=1
        corpus.append(word_tokenize(l))
    np.save(OUTPUT_NP_FILE, corpus)
#bb_ Word2Vec(corpus)

def word2vec_for_sent(corpus):
    w2v_model = Word2Vec(corpus)
    w2v_corpus = []
    for sent in corpus:
        sent_rep = []
        for word in sent:
            if word in w2v_model.vocab:
                sent_rep.append(w2v_model[word])
        w2v_corpus.append(sent_rep)
    np.save(OUTPUT_NP_W2V, w2v_corpus)
    return w2v_corpus
    

# In[]
def get_index_dict(w2v_model):
    index_dict = {}
    for word in w2v_model.vocab:
        index_dict[word]= w2v_model.vocab[word].index+1
    return index_dict

def create_model(vocab_len, batch_size, 
                 hidden_units = HIDDEN_UNITS, 
                 depth=1, 
                 impl=0
                 ):
    #input shape for one word = (100,)
    #input shape for sentence = (n, 100) n>=1, n<=70. can vary. we need to mask this.
    model = Sequential()
    model.add(
            Embedding(
                    vocab_len, 
                    batch_size, 
                    input_length=MAX_SEQ_LEN, 
                    mask_zero=True)
            )
    model.add(
            GRU(
                    hidden_units,
                    implementation=impl
                )
            )  
    model.add(RepeatVector(MAX_SEQ_LEN))
    for _ in range(depth):
        model.add(
                GRU(
                        hidden_units, 
                        return_sequences=True,  
                        implementation=impl
                    )
                )
                
    model.add(TimeDistributed(Dense(vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])    
    return model


def padinput(sequence, p_t = 'pre'):
    #100xn dimensional seq. //signle sentence
    padded_sequence = pad_sequences([sequence], MAX_SEQ_LEN, 
                                    value=PAD_VAL, 
                                    dtype='float',
                                    padding = p_t,
                                    truncating = p_t)
    return padded_sequence[0]

def prepare_data(sent, dict_size):
    freq_dist = FreqDist(np.hstack(sent))
    vocab = freq_dist.most_common(dict_size-1)
    X_ix_to_word = [word[0] for word in vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, 'ZERO')
    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    X_ix_to_word.append('UNK')
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}
    for i, sentence in enumerate(sent):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                sent[i][j] = X_word_to_ix[word]
            else:
                sent[i][j] = X_word_to_ix['UNK']
    sent = pad_sequences(sent, maxlen=MAX_SEQ_LEN, dtype='int32', value= 0)
    return sent

#sent = np.load(file)
#sent_dict = prepare_data(sent, dict_size=vocab_size)
#model = create_model(vocab_len=vocab_size+1, batch_size=train_batch_size)

def train(model, sent_dict, sent_batch_size, train_batch_size, vocab_size, start=0, end=10):
    for k in range(start,end):
        st = np.random.randint(len(sent_dict) - sent_batch_size -2)
        end = st+sent_batch_size
        
        X = sent_dict[st:end]
        Y = np.zeros((sent_batch_size, MAX_SEQ_LEN, vocab_size+1))
        for i, sentence in enumerate(sent_dict[st+1:end+1]):
            for j, word in enumerate(sentence):
                Y[i, j, word] = 1
    
        print('[INFO] Training model: epoch {}th, start:{}'.format(k, st))
        model.fit(X, Y, batch_size=train_batch_size, epochs=1, verbose=1, validation_split = 0.2)
        model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
