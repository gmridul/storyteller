
# coding: utf-8

# # Demonstrate Seq2Seq Wrapper with twitter chat log

# In[1]:

import tensorflow as tf
import numpy as np
import importlib

# preprocessed data
from dataset import data
import data_utils
importlib.reload(data_utils)


# In[2]:

# load data from pickle and npy files
metadata, idx_p, idx_x, idx_a = data.load_data(PATH='dataset/')
(trainP, trainX, trainA), (testP, testX, testA), (validP, validX, validA) = data_utils.split_dataset(idx_p, idx_x, idx_a)


# In[3]:

len(trainP), len(testX), len(testA)


# In[4]:

trainX[0]


# In[5]:

def length(x):
    for i in range(len(x)):
        if x[i] == 0:
            break
    return i

filter_index_10 = [i for i in range(len(trainX)) if length(trainX[i])==10 and length(trainP[i])>10 and length(trainA[i])>10]

filter_index_15 = [i for i in range(len(trainX)) if length(trainX[i])==15 and length(trainP[i])>10 and length(trainA[i])>10]
filter_index_07 = [i for i in range(len(trainX)) if length(trainX[i])==7 and length(trainP[i])>10 and length(trainA[i])>10]


# In[6]:

trainX_filter_10 = trainX[filter_index_10]
trainA_filter_10 = trainA[filter_index_10]
trainP_filter_10 = trainP[filter_index_10]


# In[7]:

len(filter_index_10)


# In[8]:

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainA.shape[-1]
pseq_len = trainP.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 700


# In[9]:

(small_trainX, small_trainY, small_trainP) = (trainX[:1000], trainA[:1000], trainP[:1000])
(small_testX, small_testY, small_testP) = (testX[:100], testA[:100], testP[:100]) 
(small_validX, small_validY, small_validP) = (validX[:100], validA[:100], validP[:100])


# In[10]:

import seq2seq_wrapper22


# In[11]:

importlib.reload(seq2seq_wrapper22)


# In[12]:

model = seq2seq_wrapper22.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               pseq_len=pseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt22/',
                               emb_dim=emb_dim,
                               num_layers=1,
                                epochs=1
                               )


# In[13]:

val_batch_gen = data_utils.rand_batch_gen(small_validX, small_validY, small_validP, 100)
test_batch_gen = data_utils.rand_batch_gen(small_testX, small_testY, small_testP , 100)
train_batch_gen = data_utils.rand_batch_gen(trainX_filter_10, trainA_filter_10, trainP_filter_10, batch_size)


# In[14]:

sess = model.train(train_batch_gen, val_batch_gen,)


# In[15]:

sess = model.restore_last_session()
writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)


# In[192]:

model.epochs


# In[33]:

input_ = train_batch_gen.__next__()[0]
output = model.predict(sess, input_)
print(input_.shape), print(output.shape)



# In[195]:

input_.shape


# In[199]:

story_batch_gen = data_utils.rand_batch_gen(trainX_filter_10, trainA_filter_10, 1)
input_ = story_batch_gen.__next__()[0]
for i in range(5):
    output = model.predict(sess, input_)
    for ii, oi in zip(input_.T, output):
        q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
        decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
        print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
    input_ = output.T


# In[34]:

replies = []
for ii, oi in zip(input_.T, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
    if decoded.count('unk') == 0:
        if decoded not in replies:
            #print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)


# In[31]:

tf.summary.FileWriter.add_graph(writer, graph=sess.graph)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[25]:

sess.graph


# In[ ]:



