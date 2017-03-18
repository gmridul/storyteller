# -*- coding: utf-8 -*-
# In[]
import theano
from theano import tensor
# In[]
# h_prev = h_t-1
def funnytanh(x):
    return 1.7159 * tensor.tanh(2*x/3)

def gru_encoder(x, mask, h_prev, Wr, br, Wz, bz, Ur, Uz, W, b, U):
    r = tensor.nnet.sigmoid(tensor.dot(Wr,x) + br + tensor.dot(Ur, h_prev))
    z = tensor.nnet.sigmoid(tensor.dot(Wz,x) + bz + tensor.dot(Uz, h_prev))
    h_bar = funnytanh(tensor.dot(W, x) + b + tensor.dot(U, (r * h_prev)))
    h = (1 - z) * h_prev + z * h_bar
    h = mask * h + (1-mask)*h_prev
    return h
    

def gru_decoder(x, mask, h_prev, Wr, br, Wz, bz, Ur, Uz, W, b, U, Cr, Cz, C):
    