import numpy as np
import theano
from theano import tensor
from collections import defaultdict

def shared_params(params):
    shared_p = defaultdict()
    for k, v in params.items():
        shared_p[k] = theano.shared(v, name=k)
    return shared_p

def buildModel(params):
    # shared params
    s_curr = tensor.matrix('s_curr', dtype='int32')
    s_curr_mask = tensor.matrix('s_curr_mask', dtype='float32')
    s_next = tensor.matrix('s_next', dtype='int32')
    s_next_mask = tensor.matrix('s_next_mask', dtype='float32')
    s_prev = tensor.matrix('s_prev', dtype='int32')
    s_prev_mask = tensor.matrix('s_prev_mask', dtype='float32')
    
    timesteps_curr = s_curr.shape[0]
    timesteps_next = s_next.shape[0]
    timesteps_prev = s_prev.shape[0]
    
    num_sentences = s_curr.shape[1]
    word_embeddings = params['word_emb'][s_curr.flatten()].reshape([timesteps_curr, num_sentences, params['word_dim']])
        
    # encoder layer
    encoder = gru_encoder_layer(params, word_embeddings, s_curr_mask)
    
    # decoder forward
    word_embeddings_fwd = params['word_emb'][s_next.flatten()].reshape([timesteps_next, num_sentences, params['word_dim']])
    temp = tensor.zeros_like(word_embeddings_fwd)
    word_embeddings_fwd = tensor.set_subtensor(temp[1:], word_embeddings_fwd[:-1])
    decoder_fwd = gru_decoder_layer_1(params, word_embeddings_fwd, encoder, mask, 'decoder_fwd')
    
    # decoder forward
    word_embeddings_bck = params['word_emb'][s_prev.flatten()].reshape([timesteps_prev, num_sentences, params['word_dim']])
    temp = tensor.zeros_like(word_embeddings_bck)
    word_embeddings_bck = tensor.set_subtensor(temp[1:], word_embeddings_bck[:-1])
    decoder_bck = gru_decoder_layer_1(params, word_embeddings_bck, encoder, mask, 'decoder_bck')
    
    # word probablities (fwd)
    out_linear_fwd = output_layer(params, decoder_fwd, linear, 'output_fwd')
    
    # word probablities (bck)
    out_linear_bck = output_layer(params, decoder_bck, linear, 'output_bck')
    
    