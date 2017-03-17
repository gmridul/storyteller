# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:20:16 2017

@author: kdhiman
"""
# In[]
import numpy
import gensim
import theano
import theano.tensor as tensor
# In[]
path_to_word2vec = 'GoogleNews-vectors-negative300.bin'
def load_googlenews_vectors():
    """
    load the word2vec GoogleNews vectors
    """
    embed_map = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec, binary=True)
    return embed_map
# In[]
embed_map = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec, binary=True)
# In[]
def ortho_weight(ndim):
    """
    Orthogonal weight init, for recurrent layers
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.1, ortho=True):
    """
    Uniform initalization from [-scale, scale]
    If matrix is square and ortho=True, use ortho instead
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')
# In[]
def init_tparams(params):
    """
    Initialize Theano shared variables according to the initial parameters
    """
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# In[]
params = OrderedDict()
options = OrderedDict()
options['n_words'] = 20000
options['dim_word'] = 620
params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
# In[]
x = tensor.matrix('x', dtype='int64')
tparams = init_tparams(params)
n_timesteps = x.shape[0]
n_samples = x.shape[1]
emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
f_emb = theano.function([x], emb, name='f_emb')