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
    