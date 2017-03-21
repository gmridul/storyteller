# In[]
import numpy as np
import theano
from theano import tensor
from collections import defaultdict
import layers
# In[]
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
    encoder_out = gru_encoder_layer(params, word_embeddings, s_curr_mask)
    
    # decoder forward
    decoder_fwd = gru_decoder_layer_1(params, 
                                      decoder_input(params, s_next,timesteps_next, num_sentences), 
                                      encoder_out, 
                                      mask, 'decoder_fwd')
    
    # decoder forward
    decoder_fwd = gru_decoder_layer_1(params, 
                                      decoder_input(params, s_prev, timesteps_next, num_sentences), 
                                      encoder_out, 
                                      mask, 'decoder_bck')
    
    # word probablities (fwd)
    out_linear_fwd = output_layer(params, decoder_fwd, linear, 'output_fwd')
    prob_fwd = getProb(out_linear_fwd)
    cost_fwd = getTotalCost(s_next, getProb(prob_fwd), s_next_mask)
    
    # word probablities (bck)
    out_linear_bck = output_layer(params, decoder_bck, linear, 'output_bck')
    prob_bck = getProb(out_linear_bck)
    cost_bck = getTotalCost(s_prev, getProb(prob_bck), s_prev_mask)
    
    total_cost = cost_fwd + cost_bck
    return s_curr, s_curr_mask, s_next, s_next_mask, s_nest, s_next_mask
    
def decoder_input(params, sen, timesteps_next, num_sentences):
    word_embeddings_fwd = params['word_emb'][sen.flatten()].reshape([timesteps_next, num_sentences, params['word_dim']])
    temp = tensor.zeros_like(word_embeddings_fwd)
    word_embeddings_fwd = tensor.set_subtensor(temp[1:], word_embeddings_fwd[:-1])
    return word_embeddings_fwd

def getProb(out_linear):
    return tensor.nnet.softmax(out_linear.reshape([out_linear.shape[0]*out_linear.shape[1], out_linear.shape[2]]))

def getWordProb(x, softmax_prob):
    x_flat = x.flatten() # wordids [T x S] [First words all sentces | Second words all sentcs...]
                        # T.S after flattening
                
    prob = softmax_prob.flatten(); # T.S x V [Words_Probs_s0_t0 | Words_Probs_s1_t0 ...]]
                        # T.S.V after flattening
    
    #for i in range(TS):
    #    wid = i * V + x_flat[i]
    V = softmax_prob.shape[0]
    wordids = tensor.arange(x_flat.shape[0]) * V + x_flat
    words_probs = prob[wordids]
    return words_probs.reshape(x.shape[0], x.shape[1])

def getTotalCost(x, softmax_prob, mask):
    wordProbs = getWordProb(x, softmax_prob)
    cost = - getLogLikelihood(wordProbs)
    cost = mask(cost, mask)
    return cost.sum()
    

def getLogLikelihood(prob):
    return tensor.log(prob)

def mask(x, mask):
    return x * mask
    