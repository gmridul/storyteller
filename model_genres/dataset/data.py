# In[]
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME1 = 'corpus1.txt'
FILENAME2 = 'corpus2.txt'

limit = {
        'maxq' : 30,
        'minq' : 0,
        'maxa' : 30,
        'mina' : 0
        }

label = {
         'romance' : 1,
         'horror' : -1
         }
        
UNK = 'unk'
VOCAB_SIZE = 20000

# In[]
import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle


def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return open(filename, errors='ignore').read().split('\n')[:-1]


'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''
def split_line(line):
    return line.split('.')


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(sequences):
    filtered_q, filtered_a, filtered_p = [], [], []
    raw_data_len = len(sequences)//2

    for i in range(1, len(sequences)-1, 1):
        plen, qlen, alen = len(sequences[i-1].split(' ')), len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa'] and plen >= limit['mina'] and plen <= limit['maxa']:
                filtered_p.append(sequences[i-1])
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    #print(str(filtered) + '% filtered from original data')

    return filtered_p, filtered_q, filtered_a





'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(ptokenized, qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_p = np.zeros([data_len, limit['maxa']], dtype=np.int32)
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)
    

    for i in range(data_len):
        p_indices = pad_seq(ptokenized[i], w2idx, limit['maxa'])
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_p[i] = np.array(p_indices)
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        
    return idx_p, idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data(file, genre):

    print('\n>> Read lines from file')
    lines = read_lines(filename=file)

    # change to lower case (just for en)
    lines = [ line.lower() for line in lines ]

    print('\n:: Sample from read(p) lines')
    #print(lines[121:125])

    # filter out unnecessary characters
    print('\n>> Filter lines')
    lines = [ filter_line(line, EN_WHITELIST) for line in lines ]
    #print(lines[121:125])

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    plines, qlines, alines = filter_data(lines)
    #print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
    #print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [ wordlist.split(' ') for wordlist in qlines ]
    atokenized = [ wordlist.split(' ') for wordlist in alines ]
    ptokenized = [ wordlist.split(' ') for wordlist in plines ]
    #print('\n:: Sample from segmented list of words')
    #print('\nq : {0} ; a : {1}'.format(qtokenized[60], atokenized[60]))
    #print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(ptokenized + qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    print('\n >> Zero Padding')
    idx_p, idx_q, idx_a = zero_pad(ptokenized, qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save(genre+'idx_p.npy', idx_p)
    np.save(genre+'idx_q.npy', idx_q)
    np.save(genre+'idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            genre+'w2idx' : w2idx,
            genre+'idx2w' : idx2w,
            genre+'limit' : limit,
            genre+'freq_dist' : freq_dist
        }

    # write to disk : data control dictionaries
    with open(genre+'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_data(genre,PATH=''):
    # read data control dictionaries
    with open(PATH +genre+ 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_p = np.load(PATH + genre+'idx_p.npy')
    idx_q = np.load(PATH + genre+'idx_q.npy')
    idx_a = np.load(PATH + genre+'idx_a.npy')
    return metadata, idx_p, idx_q, idx_a

# In[]
if __name__ == '__main__':
    process_data(FILENAME1, 'romance')
    process_data(FILENAME2, 'horror')
