# In[]
from collections import OrderedDict, defaultdict
from nltk.tokenize import word_tokenize
import nltk
from nltk import FreqDist
import numpy as np
import pickle as pkl
import glob
# In[]
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
def tokenize(text):
    sents = sent_detector.tokenize(text)
    token_str = ''
    # seperate each words (inclusing !, ? '', `` etc) by space
    words = []
    for s in sents:
        words += word_tokenize(s)
        
    return words
    
# In[]
def buildInverseDict(wordDict):
    inv_ =  {word:ix for ix, word in enumerate(wordDict)}
    return inv_
            
def buildDict(vocab):
    wordDict = [word[0] for word in vocab]
    wordDict.insert(0, 'ZERO')
    wordDict.append('UNK')
    return wordDict
def buildVocab(tokens, size=20000):
    words = []
    for t in tokens:
    #    words += tokenize(t)
        words += t
    dist = FreqDist(np.hstack(words))
    vocab = dist.most_common(size-1)
    return vocab
def buildVocab_Dict_InvDict(textCorpus, size=20000):
    vocab = buildVocab(textCorpus, size)
    dict_ = buildDict(vocab)
    dict_inv = buildInverseDict(dict_)
    return vocab, dict_, dict_inv
# In[]
def loadCorpus():
    corpus = np.load(OUTPUT_NP_FILE+".npy")
    return corpus

def buildAndSaveDict(textCorpus, size=20000):
    vocab, wordDict, invrWordDict = buildVocab_Dict_InvDict(textCorpus, size);
    with open(vocab_save_path+str(size)+".pkl", 'wb') as f:
        pkl.dump(vocab, f)
    with open(dict_save_path+str(size)+".pkl", 'wb') as f:
        pkl.dump(wordDict, f)
    with open(inv_dict_save_path+str(size)+".pkl", 'wb') as f:
        pkl.dump(invrWordDict, f)

def loadDict(size):
    with open(vocab_save_path+str(size)+".pkl", 'rb') as f:
        vocab = pkl.load(f)
    with open(dict_save_path+str(size)+".pkl", 'rb') as f:
        wordDict = pkl.load(f)
    with open(inv_dict_save_path+str(size)+".pkl", 'rb') as f:
        invrWordDict = pkl.load(f)
    return vocab, wordDict, invrWordDict
        
# In[]
def create_corpus(limit = 0):
    corpus = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    i=0
    for file in (glob.glob(STORY_BASE_FOLDER + '*.txt')):
        print ("Reading " + str(i))
        i+=1
        text = open(file, errors='ignore').read()
        sents = tokenizer.tokenize(text)
        
        for sent in sents[100:150]:
            sent_words = word_tokenize(sent)
            if len(sent_words) >= 2 and  len(sent_words) <= 70:
                corpus.append(sent_words)
        if i>limit and limit>0:
            break
    return corpus
      
def corpus_to_file(corpus):
    with open(OUTPUT_FILE, 'w') as f:
        for sent in corpus:
            words =  ' '.join(sent)
            f.write(words+ '\n')
            
def corpusfile_to_vec():
    corpus=[]
    sents = open(OUTPUT_FILE, errors='ignore').readlines()
    i=0
    for l in sents:
        print ("Reading " + str(i))
        i+=1
        corpus.append(word_tokenize(l))
    np.save(OUTPUT_NP_FILE, corpus)   
    
def build(size=20000):
    corpus = loadCorpus()
    buildAndSaveDict(corpus,size)
# In[]
vocab_save_path = "vocab"
dict_save_path = "word_dict"
inv_dict_save_path = "word_dict_inv"
STORY_BASE_FOLDER = './books_txt_full/Romance/'
OUTPUT_FILE = 'corpus.txt'
OUTPUT_NP_FILE = 'corpus'
#OUTPUT_NP_W2V = 'corpus_w2v'
build(5000)
vocab, wordDict, invrWordDict = loadDict(5000)