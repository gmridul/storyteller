# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:58:18 2017

@author: gopal
"""

import numpy as np
import glob
import nltk.data



# In[]
from nltk.tokenize import word_tokenize
#from nltk.corpus import brown, movie_reviews, treebank
#b = Word2Vec(brown.sents())
#b['learn']

STORY_BASE_FOLDER_ROM = '../../books_txt_full/Romance/'
STORY_BASE_FOLDER_HOR = '../../books_txt_full/Horror/'
OUTPUT_FILE = 'dataset/corpus.txt'
OUTPUT_NP_FILE = 'dataset/corpus'
OUTPUT_NP_W2V = 'dataset/corpus_w2v'


def create_corpus(base_folder, limit = 0):
    corpus = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    i=0
    for file in (glob.glob(base_folder + '*.txt')):
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

# In[]
corpus = []

corpus += create_corpus(STORY_BASE_FOLDER_ROM, limit=1000)
corpus += create_corpus(STORY_BASE_FOLDER_HOR, limit=1000)

# In[]
corpus_to_file(corpus)