# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:52:03 2017

@author: gopal
"""
import tensorflow as tf
import numpy as np
import importlib
import seq2seq_wrapper
from dataset import data
import data_utils
importlib.reload(seq2seq_wrapper)
importlib.reload(data_utils)
# preprocessed data


# load data from pickle and npy files
metadata, idx_p, idx_x, idx_a = data.load_data(PATH='dataset/')
(trainP, trainX, trainA), (testP, testX, testA), (validP, validX, validA) = data_utils.split_dataset(idx_p, idx_x, idx_a)

def length(x):
    for i in range(len(x)):
        if x[i] == 0:
            break
    return i

filter_index_10 = [i for i in range(len(trainX)) if length(trainX[i])==7 and length(trainP[i])<=10 and length(trainA[i])<=10]

trainX_filter_10 = trainX[filter_index_10]
trainA_filter_10 = trainA[filter_index_10]
trainP_filter_10 = trainP[filter_index_10]

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainA.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=1,
                                epochs=1000
                               )

sess = model.restore_last_session()


story_batch_gen = data_utils.rand_batch_gen(trainX_filter_10, trainA_filter_10, 5)
input_ = story_batch_gen.__next__()[0]
for i in range(1):
    output = model.predict(sess, input_)
    for ii, oi in zip(input_.T, output):
        q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
        decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
        print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
    input_ = output.T
