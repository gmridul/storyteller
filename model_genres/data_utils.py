import numpy as np
from random import sample

'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
def split_dataset(p, x, a, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainP, trainX, trainA = p[:lens[0]], x[:lens[0]], a[:lens[0]]
    testP, testX, testA = p[lens[0]:lens[0]+lens[1]], x[lens[0]:lens[0]+lens[1]], a[lens[0]:lens[0]+lens[1]]
    validP, validX, validA = p[-lens[-1]:], x[-lens[-1]:], a[-lens[-1]:]

    return (trainP, trainX, trainA), (testP, testX,testA), (validP, validX,validA)


'''
 generate batches from dataset
    yield (x_gen, y_gen)

    TODO : fix needed

'''
def batch_gen(x, a, p, aux, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, a[i : (i+1)*batch_size ].T, p[i : (i+1)*batch_size ].T, aux[i : (i+1)*batch_size ].T, 
'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(x, a, p, aux, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
<<<<<<< HEAD
        yield x[sample_idx].T, a[sample_idx].T, (aux[sample_idx].T).astype(float)
=======
        yield x[sample_idx].T, a[sample_idx].T, p[sample_idx].T, (aux[sample_idx].T)
>>>>>>> 0d81aa006b893195d432ed2dfbbf7f930ec62226

#'''
# convert indices of alphabets into a string (word)
#    return str(word)
#
#'''
#def decode_word(alpha_seq, idx2alpha):
#    return ''.join([ idx2alpha[alpha] for alpha in alpha_seq if alpha ])
#
#
#'''
# convert indices of phonemes into list of phonemes (as string)
#    return str(phoneme_list)
#
#'''
#def decode_phonemes(pho_seq, idx2pho):
#    return ' '.join( [ idx2pho[pho] for pho in pho_seq if pho ])


'''
 a generic decode function 
    inputs : sequence, lookup

'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])
