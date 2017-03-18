# -*- coding: utf-8 -*-
# In[]
import theano
from theano import tensor
import numpy as np
# In[]
# h_prev = h_t-1
def funnytanh(x):
    return 1.7159 * tensor.tanh(2*x/3)

def gru_encoder_logic(x, mask, h_prev, Wr, br, Wz, bz, Ur, Uz, W, b, U):
    r = tensor.nnet.sigmoid(tensor.dot(Wr,x) + br + tensor.dot(Ur, h_prev))
    z = tensor.nnet.sigmoid(tensor.dot(Wz,x) + bz + tensor.dot(Uz, h_prev))
    h_bar = funnytanh(tensor.dot(W, x) + b + tensor.dot(U, (r * h_prev)))
    h = (1 - z) * h_prev + z * h_bar
    h = mask * h + (1-mask)*h_prev
    return h
    

def gru_decoder_logic(x, mask, h_prev, Wr, br, Wz, bz, Ur, Uz, W, b, U, Cr, Cz, C, hi):
    r = tensor.nnet.sigmoid(tensor.dot(Wr,x) + br + tensor.dot(Ur, h_prev) + tensor.dot(Cr,hi))
    z = tensor.nnet.sigmoid(tensor.dot(Wz,x) + bz + tensor.dot(Uz, h_prev) + tensor.dot(Cz,hi))
    h_bar = funnytanh(tensor.dot(W, x) + b + tensor.dot(U, (r * h_prev)) + tensor.dot(C,hi))
    h = (1 - z) * h_prev + z * h_bar
    h = mask * h + (1-mask)*h_prev
    return h

def gru_decoder_logic_1(x, mask, h_prev, Wr, br, Wz, bz, Ur, Uz, W, b, U):
    r = tensor.nnet.sigmoid(tensor.dot(Wr,x) + br + tensor.dot(Ur, h_prev))
    z = tensor.nnet.sigmoid(tensor.dot(Wz,x) + bz + tensor.dot(Uz, h_prev))
    h_bar = funnytanh(tensor.dot(W, x) + b + tensor.dot(U, (r * h_prev)))
    h = (1 - z) * h_prev + z * h_bar
    h = mask * h + (1-mask)*h_prev
    return h

def gru_encoder_layer(params, x, mask):
    seqs = [x, mask]
    p = params['encoder']
    init = tensor.alloc(0., x.shape[1], params['num_hidden'])
    outputs, updates = theano.scan(gru_encoder_logic, 
                                   sequences = seqs, 
                                   output_info = [init],
                                   non_sequences=[p['Wr'], p['br'], p['Wz'], 
                                                  p['bz'], p['Ur'], p['Uz'], 
                                                  p['W'], p['b'], p['U']],
                                   name='encoder',
                                   n_steps=x.shape[0],
                                   strict=True
                                )
    return outputs[-1]
    
def gru_decoder_layer(params, x, hi, mask, dec_type):
    seqs = [x, mask]
    p = params[dec_type]
    init = tensor.alloc(0., x.shape[1], params['num_hidden'])
    outputs, updates = theano.scan(gru_decoder_logic, 
                                   sequences = seqs, 
                                   output_info = [init],
                                   non_sequences=[p['Wr'], p['br'], p['Wz'], 
                                                  p['bz'], p['Ur'], p['Uz'], 
                                                  p['W'], p['b'], p['U'], 
                                                  p['Cr'], p['Cz'],p['C'], hi],
                                   name='encoder',
                                   n_steps=x.shape[0],
                                   strict=True
                                )
    return outputs[-1]

def gru_decoder_layer_1(params, x, hi, mask, dec_type):
    seqs = [x, mask]
    p = params[dec_type]
    init = hi
    outputs, updates = theano.scan(gru_decoder_logic_1, 
                                   sequences = seqs, 
                                   output_info = [init],
                                   non_sequences=[p['Wr'], p['br'], p['Wz'], 
                                                  p['bz'], p['Ur'], p['Uz'], 
                                                  p['W'], p['b'], p['U'], 
                                                  hi],
                                   name='encoder',
                                   n_steps=x.shape[0],
                                   strict=True
                                )
    return outputs[-1]

def orthoInit(dim):
    t = numpy.random.randn(dim, dim)
    w, _t, _t = numpy.linalg.svd(t)
    return w.astype('float32')

def initial_weights(dim1, dim2, scale = 0.1):
    if dim1==dim2:
        wis = orthoInit(dim1)
    else:
        wis = numpy.random.uniform(low=-scale, high=scale, size=(dim1, dim2))
    return wis.astype('float32')

def init_encoder_params(params):
    word_dim = params['word_dim']
    num_hidden = params['num_hidden']
    params['encoder']['W']  = theano.shared(initial_weights(word_dim, num_hidden))
    params['encoder']['Wr'] = theano.shared(initial_weights(word_dim, num_hidden))
    params['encoder']['Wz'] = theano.shared(initial_weights(word_dim, num_hidden))
    
    params['encoder']['U']  = theano.shared(initial_weights(num_hidden, num_hidden))
    params['encoder']['Ur'] = theano.shared(initial_weights(num_hidden, num_hidden))
    params['encoder']['Uz'] = theano.shared(initial_weights(num_hidden, num_hidden))
    
    params['encoder']['b']  = theano.shared(np.zeros(num_hidden))
    params['encoder']['br'] = theano.shared(np.zeros(num_hidden))
    params['encoder']['bz'] = theano.shared(np.zeros(num_hidden))
    return params

def init_decoder_params(params):
    word_dim = params['word_dim']
    num_hidden = params['num_hidden']
    params['decoder_fwd']['W']  = theano.shared(initial_weights(word_dim, num_hidden))
    params['decoder_fwd']['Wr'] = theano.shared(initial_weights(word_dim, num_hidden))
    params['decoder_fwd']['Wz'] = theano.shared(initial_weights(word_dim, num_hidden))
    
    params['decoder_fwd']['U']  = theano.shared(initial_weights(num_hidden, num_hidden))
    params['decoder_fwd']['Ur'] = theano.shared(initial_weights(num_hidden, num_hidden))
    params['decoder_fwd']['Uz'] = theano.shared(initial_weights(num_hidden, num_hidden))
    
    params['decoder_fwd']['b']  = theano.shared(np.zeros(num_hidden))
    params['decoder_fwd']['br'] = theano.shared(np.zeros(num_hidden))
    params['decoder_fwd']['bz'] = theano.shared(np.zeros(num_hidden))
    
    
    params['decoder_bck']['W']  = theano.shared(initial_weights(word_dim, num_hidden))
    params['decoder_bck']['Wr'] = theano.shared(initial_weights(word_dim, num_hidden))
    params['decoder_bck']['Wz'] = theano.shared(initial_weights(word_dim, num_hidden))
    
    params['decoder_bck']['U']  = theano.shared(initial_weights(num_hidden, num_hidden))
    params['decoder_bck']['Ur'] = theano.shared(initial_weights(num_hidden, num_hidden))
    params['decoder_bck']['Uz'] = theano.shared(initial_weights(num_hidden, num_hidden))
    
    params['decoder_bck']['b']  = theano.shared(np.zeros(num_hidden))
    params['decoder_bck']['br'] = theano.shared(np.zeros(num_hidden))
    params['decoder_bck']['bz'] = theano.shared(np.zeros(num_hidden))
    
    return params

def init_output_layer_params(params):
    num_hidden = params['num_hidden']
    vocab_size = params['vocab_size']
    params['output_fwd']['W'] = theano.shared(initial_weights(num_hidden, vocab_size))
    params['output_fwd']['b'] = theano.shared(np.zeros(vocab_size))
    
    params['output_bck']['W'] = theano.shared(initial_weights(num_hidden, vocab_size))
    params['output_bck']['b'] = theano.shared(np.zeros(vocab_size))
    
    return params

def output_layer(params, x, activation, dec_type):
    return activation(tensor.dot(params[dec_type]['W'], x) + params[dec_type]['b'])

def linear(x):
    return x

def softmax(x):
    