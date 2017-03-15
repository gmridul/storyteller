# In[]
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.recurrent import GRU
from keras.layers.core import RepeatVector, Masking
from keras.preprocessing.sequence import pad_sequences

# In[]
GRU_OUTPUT_DIM = 2400
MAX_SEQ_LEN = 50
PAD_VAL = -np.inf
# In[]
def padinput(sequence):
    padded_sequence = pad_sequences(sequence, MAX_SEQ_LEN, value=PAD_VAL)
    # padding and truncating is 'pre'
    return padded_sequence

# In[]
def funnytanh(x):
    return 1.7159*np.tanh(x*2/3)

# In[]
adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

def GRU_encode():
    return GRU(
               units=GRU_OUTPUT_DIM, 
               activation=funnytanh, 
               return_sequence=False
              ) #more arguments to add
    
def GRU_decode():
    return GRU(
               units=GRU_OUTPUT_DIM, 
               activation=funnytanh, 
               return_sequence=True
               ) #more arguments to add

def mask_layer():
    return  Masking(mask_value=PAD_VAL)

# MODEL
model = Sequential()
model.add(mask_layer())
model.add(GRU_encode())
model.add(RepeatVector(MAX_SEQ_LEN))
model.add(GRU_decode())
