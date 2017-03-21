# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:20:16 2017

@author: kdhiman
"""
# In[]
import numpy as np
import gensim
import theano
import theano.tensor as tensor
from scipy.linalg import norm
import tools
from collections import defaultdict, OrderedDict
from sklearn.linear_model import LinearRegression

# In[] = 
google_word_to_vec_path = "GoogleNews-vectors-negative300.bin"
googleWord2vec = gensim.models.KeyedVectors.load_word2vec_format(google_word_to_vec_path, binary=True)
# In[]
def wordEmbeddingRandomUniform(num_word, word_dim, scale):
    if(word_dim==num_word):
        print("Warn: word_dim = num_word! Try ortho initilization")
    return np.random.uniform(low=-scale, high=scale, size=(num_word, word_dim))
# In[]
#def getTheanoParams(params):
#    dict_ = defaultdict()
#    for k,v in dict_.items():
#        dict_[k] = theano.shared(v)
#    return dict_
# In[]
num_words = 5000
word_dim = 620
# In[]
vocab, wordIds, idsToWord = tools.loadDict(num_words)
# In[]
#tparam = getTheanoParams(params)
# Initialize word embedding matrix -- random initiliazation 
wordEmb = wordEmbeddingRandomUniform(num_words, word_dim, 0.1)
# In[]
#x = tensor.matrix('x', dtype='int64')     
#max_seq_len = x.shape[0]
#num_sents = x.shape[1]
#emb_ = wordEmb[x.flatten()].reshape([max_seq_len, num_sents, params['word_dim']])
# theano function to get word embddings 
# dimentions:: (sentence length, #sentences, word dimention)
#fun_emb = theano.function([x], emb_, name='f_emb')
# In[]
# Linear mapping
X = [] #google_word2vec space
y = [] # rrn space

for wordId in range(num_words-2):
    print(wordId)
    word = idsToWord[wordId]
    if word in googleWord2vec:
        X.append(googleWord2vec[word])
        y.append(wordEmb[wordId])
    
# In[]
clf = LinearRegression()
clf.fit(X,y)

# In[]
wordEmbExtended = defaultdict()
for word in wordIds:
    wordEmbExtended[word] = wordEmb[wordIds[word]]

count = 1000
new_words = []
for word in googleWord2vec.vocab:
    if '_' not in word and '.' not in word and '@' not in word:
        if word not in wordIds:
            if not word[0].isupper():
                new_words.append(word)
                wordEmbExtended[word] = clf.predict([googleWord2vec[word]])
                if(count==0):
                    break;
                count -= 1
        

# In[]
# In[]
#e = wordEmb[0]
#e = e.reshape(word_dim,1)
#c = np.dot(fair.T, e)/(norm(e) * norm(fair))
    
# In[]
fair = wordEmbExtended['actively'].reshape(word_dim,1)
cos_sim = 0
index = -1
for i in range(len(wordEmb)):
    e = wordEmb[i]
    e = e.reshape(word_dim,1)
    c = np.dot(fair, e.T)/(norm(e) * norm(fair))
    if c[0][0] > cos_sim:
        index = i
        cos_sim = c[0][0]
print(idsToWord[index])