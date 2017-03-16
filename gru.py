# In[]
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.recurrent import GRU,LSTM
from keras.layers.core import RepeatVector, Masking
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
# In[Macros]
GRU_OUTPUT_DIM = 2400
MAX_SEQ_LEN = 50
PAD_VAL = 0
VEC_LEN = 100
# In[]
def padinput(sequence):
    #100xn dimensional seq. 
    padded_sequence = pad_sequences([sequence], MAX_SEQ_LEN, value=PAD_VAL)
    # padding and truncating is 'pre'
    return padded_sequence[0]

# In[]
def funnytanh(x):
    return 1.7159*np.tanh(x*2/3)

# In[]
 
adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

def GRU_encode():
    return GRU(32,
               units=GRU_OUTPUT_DIM, 
               activation=funnytanh, 
               input_dim=(MAX_SEQ_LEN, VEC_LEN),
               output_dim=(),
               return_sequence=False
              ) #more arguments to add
    
def GRU_decode():
    return GRU(32,
               units=GRU_OUTPUT_DIM, 
               activation=funnytanh, 
               return_sequence=True
               ) #more arguments to add

def mask_layer():
    return  Masking(mask_value=PAD_VAL, input_shape=(None, VEC_LEN))

# In[MODEL]
model = Sequential()
model.add(Masking(mask_value=PAD_VAL, input_shape=(MAX_SEQ_LEN,VEC_LEN)))

# In[]
model.add(GRU( 
              units=GRU_OUTPUT_DIM, 
              input_shape=(VEC_LEN,MAX_SEQ_LEN), 
              return_sequences=False, 
              stateful=False
             ))


# In[]               #input_dim=(MAX_SEQ_LEN, VEC_LEN),
               
               #return_sequence=False

#model.add(RepeatVector(MAX_SEQ_LEN))
#model.add(GRU( units=GRU_OUTPUT_DIM,
             #  return_sequences=True, 
             #  stateful=True 
             # ))
               #return_sequence=True


