
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

from collections import defaultdict
# In[9]:
def length(x):
	for i in range (len(x)):
		if x[i] == 0 :
			break
	return i

def filter10(p,c,n):
     return length(c)==10 and length(p)>=7 and length(n)>=7

def AaneDo(*arg):
    return True

def filterit(c, p, n, label, howmany, filter_func):
    return [[p[i], c[i], n[i], np.full(len(n[0]),label, dtype=int)] for i in range(howmany) if filter_func(p[i], c[i], n[i])]
    
# load data from pickle and npy files
def getdata(genre):
    metadata, idx_prev, idx_curr, idx_next = data.load_data(genre,PATH='dataset/')
    (train_prev, train_curr, train_next), (test_prev, test_curr, test_next), (valid_prev, valid_curr, valid_next) \
        = data_utils.split_dataset(idx_prev, idx_curr, idx_next)
    train = defaultdict()
    train['p'] = train_prev
    train['c'] = train_curr
    train['n'] = train_next
    
    valid = defaultdict()
    valid['p'] = valid_prev
    valid['c'] = valid_curr
    valid['n'] = valid_next
    
    test = defaultdict()
    test['p'] = test_prev
    test['c'] = test_curr
    test['n'] = test_next
    
    return train, valid, test, metadata
    # In[12]:
    
    #len(test_prev), len(test_curr), len(test_next)
    
    # In[14]:
    
    # parameters 
    
    # In[22]:
    #==10, >=7, >=7
def filter_data(train, valid, test, label, filter_func):
    train_data = filterit(train['p'], train['c'], train['n'], label, len(train['c']), filter_func)
    valid_data = filterit(valid['p'], valid['c'], valid['n'], label, 100, AaneDo)
    test_data = filterit(test['p'], test['c'], test['n'], label, 100, AaneDo)
    return np.array(train_data), np.array(valid_data), np.array(test_data)
    
# In[17]
ids = {'train':0, 'valid':1, 'test':2, 'metadata':3,'prev':0, 'curr':1, 'next':2, 'label':3}
romance_data = getdata("romance")
horror_data = getdata("horror")
label_rom, label_hor = 1, -1
hor_metadata = horror_data[ids['metadata']]
rom_metadata = romance_data[ids['metadata']]
rom_fil_data = filter_data(romance_data[0], romance_data[1], romance_data[2], label_rom, filter10)
hor_fil_data = filter_data(horror_data[0],  horror_data[1],  horror_data[2],  label_hor, filter10)
# In[]


rom_train_data = rom_fil_data[ids['train']]
hor_train_data = hor_fil_data[ids['train']]
train_data = np.concatenate((rom_train_data, hor_train_data), axis=0)
np.random.shuffle(train_data)

rom_valid_data = rom_fil_data[ids['valid']]
hor_valid_data = hor_fil_data[ids['valid']]
valid_data = np.concatenate((rom_valid_data, hor_valid_data), axis=0)
np.random.shuffle(valid_data)


xseq_len = rom_fil_data[ids['train']][0][ids['curr']].shape[-1]
yseq_len = rom_fil_data[ids['train']][0][ids['next']].shape[-1]
zseq_len = rom_fil_data[ids['train']][0][ids['prev']].shape[-1]
batch_size = 64
xvocab_size =  len(horror_data[ids['metadata']]['horroridx2w'])  
yvocab_size = xvocab_size
zvocab_size = xvocab_size
emb_dim = 1536

# In[]
import seq2seq_wrapper

# In[23]:

importlib.reload(seq2seq_wrapper)


# In[47]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               zseq_len=zseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               zvocab_size=zvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=1,
                                epochs=5
                               )

# In[21]:

val_batch_gen = data_utils.rand_batch_gen(valid_data[:,1,:], valid_data[:,2,:], valid_data[:,0,:], valid_data[:,3,:], 100)
#test_batch_gen = data_utils.rand_batch_gen(small_testX, small_testY, small_testAx, 100)
train_batch_gen = data_utils.rand_batch_gen(train_data[:,1,:], train_data[:,2,:], train_data[:,0,:], train_data[:,3,:], batch_size)
#train_batch_gen_story = data_utils.rand_batch_gen(trainX_filter_10, trainN_filter_10, trainAx_filter_10, 1)

# In[ ]:
#sess = model.restore_last_session()
#sess = model.train(train_batch_gen, val_batch_gen, sess)
if(False):
        sess = model.restore_last_session()
        sess = model.train(train_batch_gen, val_batch_gen, sess)
else:
        sess = model.train(train_batch_gen, val_batch_gen, )

# In[25]:

sess = model.restore_last_session()


# In[43]:

model.epochs


# In[43]:

input_ = train_batch_gen.__next__()[0]
input_aux_ = train_batch_gen.__next__()[3]
output, output_bwd = model.predict(sess, input_, input_aux_)
print(output.shape)


# In[44]:


replies = []
for ii, ai, oi, oi_bwd in zip(input_.T, input_aux_.T, output, output_bwd):
    genre = ''
    
    if ai[0] > 0:
        genre = 'romance'
        lookup_ = rom_metadata
    else:
        genre = 'horror'
        lookup_ = hor_metadata
        
    q = data_utils.decode(sequence=ii, lookup=lookup_[genre+'idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=lookup_[genre+'idx2w'], separator=' ').split(' ')
    decoded_bwd = data_utils.decode(sequence=oi_bwd, lookup=lookup_[genre+'idx2w'], separator=' ').split(' ')
    print('q : [{0}]; a : [{1}]; p : [{2}]'.format(q, ' '.join(decoded), ' '.join(decoded_bwd)))
    if decoded.count('unk') == 0:
        if decoded not in replies:
            #print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)

# In[ ]:


#input_ = val_batch_gen.__next__()[0]
#output = model.predict(sess, input_)
#print(output.shape)


# In[44]:




#for ii, oi in zip(input_.T, output):
#	q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
#	decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
#	print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
    	