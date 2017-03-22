
# coding: utf-8

# # Demonstrate Seq2Seq Wrapper with twitter chat log

# In[20]:

import tensorflow as tf
import numpy as np
import importlib

# preprocessed data
from dataset import data
import data_utils
importlib.reload(data_utils)


# In[9]:

# load data from pickle and npy files
metadata, idx_p, idx_x, idx_a = data.load_data(PATH='dataset/')
(trainP, trainX, trainA), (testP, testX, testA), (validP, validX, validA) = data_utils.split_dataset(idx_p, idx_x, idx_a)


# In[12]:

len(testP), len(testX), len(testA)

# In[14]:

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainA.shape[-1]
batch_size = 4
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 512

# In[22]:
def length(x):
	for i in range (len(x)):
		if x[i] == 0 :
			break
	return i

filter_index_10 = [i for i in range(len(trainX)) if length(trainX[i])>=3 and length(trainP[i])>=3 and length(trainA[i])>=3]

trainX_filter_10 = trainX[filter_index_10]
trainA_filter_10 = trainA[filter_index_10]
trainP_filter_10 = trainP[filter_index_10]
trainAx_filter_10 = np.zeros((trainA_filter_10.shape[0], trainA_filter_10.shape[1]), dtype=float)

# In[15]:
num_sent = 10
(small_trainX, small_trainY) = (trainX[:num_sent], trainA[:num_sent])
(small_testX, small_testY) = (testX[:100], testA[:100]) 
(small_validX, small_validY) = (validX[:100], validA[:100])
small_trainAx =np.zeros((small_trainY.shape[0], small_trainY.shape[1]), dtype=float)
small_testAx = np.zeros((small_testY.shape[0], small_testY.shape[1]), dtype=float)
small_validAx = np.zeros((small_validY.shape[0], small_validY.shape[1]), dtype=float)

# In[17]:

import seq2seq_wrapper


# In[23]:

importlib.reload(seq2seq_wrapper)


# In[47]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=1,
                                epochs=5
                               )


# In[21]:

val_batch_gen = data_utils.rand_batch_gen(small_validX, small_validY, small_validAx, 100)
test_batch_gen = data_utils.rand_batch_gen(small_testX, small_testY, small_testAx, 100)
train_batch_gen = data_utils.rand_batch_gen(trainX_filter_10, trainA_filter_10, trainAx_filter_10, batch_size)
train_batch_gen_story = data_utils.rand_batch_gen(trainX_filter_10, trainA_filter_10, trainAx_filter_10, 1)

# In[ ]:
#sess = model.restore_last_session()
#sess = model.train(train_batch_gen, val_batch_gen, sess)
sess = model.train(train_batch_gen, val_batch_gen, )

# In[25]:

#sess = model.restore_last_session()


# In[43]:

model.epochs


# In[43]:

input_ = train_batch_gen.__next__()[0]
output = model.predict(sess, input_)
print(output.shape)


# In[44]:


replies = []
for ii, oi in zip(input_.T, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
    if decoded.count('unk') == 0:
        if decoded not in replies:
            #print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)

# In[ ]:




# In[ ]:


#input_ = val_batch_gen.__next__()[0]
#output = model.predict(sess, input_)
#print(output.shape)


# In[44]:




#for ii, oi in zip(input_.T, output):
#	q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
#	decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
#	print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
    	
# In[ ]:

# In[ ]:




# In[ ]:




# In[ ]:



